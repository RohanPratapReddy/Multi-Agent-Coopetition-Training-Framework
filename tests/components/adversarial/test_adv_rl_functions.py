"""Adversarial verification of D2 for ``rl_function`` components (doc 17 section 8).

The compliance charge belongs to the *framework's* loss terms. A custom loss cannot claim to
carry it (``spec_type`` / a ``terms`` attribute), a custom rollup cannot drop a child's void set,
and a function that is not deterministic, not finite or edits its inputs is refused at build.
"""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_function
from mactf.components.loss_compliance import carries_compliance, ensure_compliance
from mactf.exceptions import ComplianceGuardViolation, ComponentContractError, ImproperlyConfigured
from mactf.rl.functions.loss import ComplianceFloorLoss, CompositeLoss, PolicyGradientLoss, WeightedSumLoss
from mactf.rl.loss import LossBreakdown
from mactf.scoring.base import ScorerOutput, ScoringKind

from adv_support import breach_trainer, run_epochs
from p2_support import configure, make_trainer, tune


class _Ignores:
    """A loss that ignores the compliance floor and claims otherwise."""

    required_fields = frozenset({"values", "value_targets"})
    spec_type = "compliance_floor"                       # the spoof
    terms = [ComplianceFloorLoss()]                      # a decoy it never calls

    def compute(self, batch):
        loss = (batch["values"] - batch["value_targets"].detach()).pow(2).mean()
        return loss, LossBreakdown(total=float(loss.detach()))

    def describe(self):
        return {"type": "ignores"}


class _ClaimsComposite(_Ignores):
    spec_type = "composite"


def test_a_loss_cannot_claim_to_carry_the_compliance_term():
    for loss in (_Ignores(), _ClaimsComposite()):
        assert not carries_compliance(loss)
        extended = ensure_compliance(loss, mode="require", coefficient=1.0)
        assert isinstance(extended, WeightedSumLoss) and carries_compliance(extended)
        with pytest.raises(Exception, match="no compliance_floor term"):
            ensure_compliance(loss, mode="enforce", coefficient=1.0)


def test_the_framework_terms_still_count_and_a_wrapper_with_a_decoy_does_not():
    assert carries_compliance(CompositeLoss()) and carries_compliance(ComplianceFloorLoss())
    assert carries_compliance(WeightedSumLoss([PolicyGradientLoss(), ComplianceFloorLoss()], weights=[1.0, 1.0]))
    assert not carries_compliance(PolicyGradientLoss())

    class Wrapper(_Ignores):
        spec_type = "chained"
        terms = [ComplianceFloorLoss()]

    assert not carries_compliance(Wrapper())


def test_a_configured_ignoring_loss_is_extended_in_a_real_trainer(settings, tmp_path, component_overlay):
    register_function("loss", "ignores", _Ignores, scope="run")
    tune(settings)
    configure(settings, RL={"LOSS": {"type": "ignores"}})
    trainer = make_trainer(settings, tmp_path).build()
    loss = trainer.loss_for("peer-1")
    assert isinstance(loss, WeightedSumLoss) and carries_compliance(loss)
    values = torch.randn(4, requires_grad=True)
    batch = {"values": values, "value_targets": values.detach().clone(), "compliance_floor": torch.tensor(-2.0)}
    total, breakdown = loss.compute(batch)
    assert breakdown.compliance > 0.0 and float(total.detach()) >= 2.0       # the charge for a floor of -2


# -- functions refused at build ---------------------------------------------------------------
class _NaNAdvantage:
    def __call__(self, rewards, values, **kw):
        from mactf.rl.advantage import AdvantageBatch

        bad = torch.full_like(rewards, float("nan"))
        return AdvantageBatch(advantages=bad, returns=bad)


class _MutatesInputs:
    def __call__(self, rewards, values, **kw):
        from mactf.rl.advantage import AdvantageBatch

        rewards.zero_()
        return AdvantageBatch(advantages=rewards.clone(), returns=rewards.clone())


class _Random:
    def __call__(self, rewards, values, **kw):
        from mactf.rl.advantage import AdvantageBatch

        noise = torch.rand_like(rewards)
        return AdvantageBatch(advantages=noise, returns=noise)


@pytest.mark.parametrize("cls", [_NaNAdvantage, _MutatesInputs, _Random])
def test_a_broken_advantage_function_is_refused_at_build(cls, settings, tmp_path, component_overlay):
    cls.spec_family = "advantage"
    cls.spec_type = cls.__name__.lower()
    register_function("advantage", cls.spec_type, cls, scope="run")
    tune(settings)
    configure(settings, RL={"ADVANTAGE": {"type": cls.spec_type}})
    with pytest.raises((ImproperlyConfigured, ComponentContractError), match="F00[4-7]|probe|F004"):
        make_trainer(settings, tmp_path).build()


# -- rollup ------------------------------------------------------------------------------------
class _DropsVoid:
    spec_family = "rollup"
    name = "drops_void"

    def rollup(self, parent, children, *, weights=None, scores=None):
        return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=sum(c.amount for c in children))

    def describe(self):
        return {"type": "drops_void"}


def test_a_rollup_that_drops_a_childs_void_set_is_refused_at_build(settings, tmp_path, component_overlay):
    register_function("rollup", "drops_void", _DropsVoid, scope="run")
    from p2_support import hierarchy_trainer

    tune(settings)
    configure(settings, RL={"ROLLUP": {"type": "drops_void"}})
    with pytest.raises((ImproperlyConfigured, ComponentContractError), match="F010|void"):
        hierarchy_trainer(settings, tmp_path).build()


def test_a_rollup_that_drops_void_at_run_time_is_caught_by_the_guard():
    from mactf.components.guards.rollup import RollupGuard

    guard = RollupGuard(_DropsVoid(), mode="enforce")
    kids = (ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1.0, void_rules=("P1",)),)
    with pytest.raises(ComplianceGuardViolation) as caught:
        guard.rollup("root", kids)
    assert caught.value.code == "C010"
    fallback = RollupGuard(_DropsVoid(), mode="fallback").rollup("root", kids)
    assert "P1" in fallback.void_rules


def test_a_rollup_output_is_read_once():
    from mactf.components.guards.rollup import RollupGuard

    class Lazy:
        spec_family = "rollup"
        name = "lazy"

        def rollup(self, parent, children, *, weights=None, scores=None):
            class Once:
                used = False

                def __iter__(self_inner):
                    if Once.used:
                        return iter(())
                    Once.used = True
                    return iter(("P1",))

            return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1.0, void_rules=Once())

    kids = (ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1.0, void_rules=("P1",)),)
    out = RollupGuard(Lazy(), mode="enforce").rollup("root", kids)
    assert tuple(out.void_rules) == ("P1",) and tuple(out.void_rules) == ("P1",)


@pytest.mark.parametrize("amount", ["5.0", float("nan"), float("inf"), None, True])
def test_a_rollup_amount_that_is_not_a_finite_number_is_refused(amount):
    from mactf.components.guards.rollup import RollupGuard

    class Bad:
        spec_family = "rollup"
        name = "bad"

        def rollup(self, parent, children, *, weights=None, scores=None):
            return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=amount)

    with pytest.raises(ComplianceGuardViolation) as caught:
        RollupGuard(Bad(), mode="enforce").rollup("root", ())
    assert caught.value.code == "C011"
