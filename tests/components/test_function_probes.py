"""``verify_function``: one deliberately broken function per finding code (section 8.3)."""

from __future__ import annotations

import pytest
import torch

from mactf.components.function_probes import (
    EXEMPTABLE,
    PROBES,
    ProbeContext,
    check_function,
    verify_function,
)
from mactf.exceptions import ComponentContractError
from mactf.rl.advantage import AdvantageBatch
from mactf.rl.functions.advantage import GAEAdvantage
from mactf.rl.functions.loss import CompositeLoss
from mactf.rl.functions.rollup import KappaWeightedRollup
from mactf.rl.functions.support import ClipGradient, ConsensusSynthesis, GreedyAction
from mactf.rl.loss import LossBreakdown
from mactf.scoring.base import ScorerOutput, ScoringKind

from p2_support import ClippedAdvantage, HuberValueLoss, L1AnnealRegularization, MedianSynthesis


def codes(findings):
    return sorted({finding.code for finding in findings})


# -- the well-behaved functions pass -------------------------------------------------------------------
@pytest.mark.parametrize("family,instance", [
    ("advantage", ClippedAdvantage(clip=1.0)),
    ("advantage", GAEAdvantage()),
    ("loss", HuberValueLoss()),
    ("loss", CompositeLoss()),
    ("regularization", L1AnnealRegularization()),
    ("rollup", KappaWeightedRollup()),
    ("gradient", ClipGradient()),
    ("action", GreedyAction()),
    ("synthesis", MedianSynthesis()),
    ("synthesis", ConsensusSynthesis()),
])
def test_well_behaved_functions_have_no_findings(family, instance):
    assert verify_function(family, instance) == []


def test_every_family_has_a_probe():
    from mactf.components.kinds import RL_FAMILIES

    assert set(PROBES) == set(RL_FAMILIES)


# -- F001 .. F012 -----------------------------------------------------------------------------------------
class Raises:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("no")


def test_f001_the_entry_method_raised():
    findings = verify_function("advantage", Raises())
    assert codes(findings) == ["F001"] and "RuntimeError" in findings[0].message


def test_f002_wrong_return_type():
    class Strings:
        def __call__(self, rewards, values, **kwargs):
            return "advantages"

    assert codes(verify_function("advantage", Strings())) == ["F002"]
    class Pair:
        def compute(self, batch):
            return 1.0

    assert codes(verify_function("loss", Pair())) == ["F002"]


def test_f003_wrong_shape():
    class Squared:
        def __call__(self, rewards, values, **kwargs):
            return AdvantageBatch(advantages=rewards.unsqueeze(0).expand(rewards.numel(), -1).reshape(-1),
                                  returns=rewards)

    assert codes(verify_function("advantage", Squared())) == ["F003"]


def test_f004_non_finite_output_and_breakdown():
    class Nan:
        def __call__(self, rewards, values, **kwargs):
            return AdvantageBatch(advantages=rewards * float("nan"), returns=rewards)

    assert "F004" in codes(verify_function("advantage", Nan()))

    class BadBreakdown(HuberValueLoss):
        def compute(self, batch):
            loss, parts = super().compute(batch)
            return loss, LossBreakdown(kl=float("inf"), total=float(loss.detach()))

    assert "F004" in codes(verify_function("loss", BadBreakdown()))


def test_f005_not_differentiable():
    class Detached(HuberValueLoss):
        def compute(self, batch):
            loss, parts = super().compute(batch)
            return loss.detach(), parts

    assert codes(verify_function("loss", Detached())) == ["F005"]


def test_f005_a_regularizer_not_connected_to_the_module():
    class Constant:
        def __call__(self, module):
            return torch.tensor(1.0, requires_grad=True)

    assert codes(verify_function("regularization", Constant())) == ["F005"]


def test_f006_not_deterministic():
    class Noisy(HuberValueLoss):
        def compute(self, batch):
            loss, _ = super().compute(batch)
            noisy = loss + torch.rand(())
            return noisy, LossBreakdown(total=float(noisy.detach()))

    assert codes(verify_function("loss", Noisy())) == ["F006"]


def test_f007_mutates_an_input():
    class Mutator:
        def __call__(self, rewards, values, **kwargs):
            rewards.add_(1.0)
            return AdvantageBatch(advantages=rewards, returns=rewards)

    assert "F007" in codes(verify_function("advantage", Mutator()))

    class BatchMutator(HuberValueLoss):
        def compute(self, batch):
            batch["values"].data.add_(1.0)
            return super().compute(batch)

    assert "F007" in codes(verify_function("loss", BatchMutator()))


def test_f008_breakdown_total_disagrees_with_the_loss():
    class Liar(HuberValueLoss):
        def compute(self, batch):
            loss, _ = super().compute(batch)
            return loss, LossBreakdown(total=123.0)

    assert codes(verify_function("loss", Liar())) == ["F008"]


def test_f009_a_loss_that_needs_keys_the_probe_cannot_supply():
    class Needy(HuberValueLoss):
        required_fields = frozenset({"mystery"})

    findings = verify_function("loss", Needy())
    assert codes(findings) == ["F009"] and "probe_batch" in findings[0].message

    class Supplied(HuberValueLoss):
        required_fields = frozenset({"mystery"})

        @staticmethod
        def probe_batch(ctx):
            values = torch.zeros(ctx.batch, 1, requires_grad=True)
            return {"values": values, "value_targets": torch.ones(ctx.batch, 1), "mystery": 1}, (values,)

    assert verify_function("loss", Supplied()) == []


def test_f010_a_rollup_that_drops_a_childs_void_set():
    class Forgetful:
        def rollup(self, parent, children, *, weights=None, scores=None):
            return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=float(sum(c.amount for c in children)))

    findings = verify_function("rollup", Forgetful())
    assert codes(findings) == ["F010"] and "P1" in findings[0].message


def test_f011_only_three_codes_can_be_skipped():
    assert EXEMPTABLE == {"F005", "F006", "F007"}

    class Skips(HuberValueLoss):
        probe = {"skip": ("F004", "F006")}

    findings = verify_function("loss", Skips())
    assert codes(findings) == ["F011"] and "F004" in findings[0].message


def test_a_declared_skip_silences_exactly_that_code():
    class Noisy(HuberValueLoss):
        probe = {"skip": ("F006",)}

        def compute(self, batch):
            loss, _ = super().compute(batch)
            noisy = loss + torch.rand(())
            return noisy, LossBreakdown(total=float(noisy.detach()))

    assert verify_function("loss", Noisy()) == []


def test_f012_synthesis_shape_rule():
    class Reshaper:
        def __call__(self, proposals, *, weights=None):
            return torch.cat(list(proposals.values()), dim=-1)

    assert codes(verify_function("synthesis", Reshaper())) == ["F012"]

    class Declared(Reshaper):
        output_shape_rule = "free"

    assert verify_function("synthesis", Declared()) == []


def test_action_must_return_integer_class_indices():
    class Floats:
        def __call__(self, logits, *, sample):
            return logits.argmax(-1).float()

    assert codes(verify_function("action", Floats())) == ["F002"]

    class OutOfRange:
        def __call__(self, logits, *, sample):
            return torch.full((logits.shape[0],), logits.shape[1])

    assert codes(verify_function("action", OutOfRange())) == ["F003"]


def test_a_gradient_function_may_not_change_parameter_values():
    class Zaps:
        def __call__(self, parameters, *, max_norm, scaler):
            for parameter in parameters:
                parameter.data.zero_()
            return 0.0

    assert "F007" in codes(verify_function("gradient", Zaps()))


def test_metric_values_must_be_numbers():
    class Strings:
        def __call__(self, batch, output):
            return {"accuracy": "high"}

    assert codes(verify_function("metric", Strings())) == ["F002"]


def test_reward_must_be_a_finite_number():
    class Nan:
        def __call__(self, episode, *, peer):
            return float("nan")

    assert "F004" in codes(verify_function("reward", Nan()))


def test_return_function_shape():
    class Short:
        def __call__(self, rewards, *, gamma, masks=None):
            return rewards[:-1]

    assert codes(verify_function("return", Short())) == ["F003"]


def test_value_function_needs_a_value_estimate():
    class Tensor:
        def __call__(self, states, **kwargs):
            return states.sum(-1)

    assert codes(verify_function("value", Tensor())) == ["F002"]


def test_probe_sizes_can_be_adjusted_by_the_function():
    class Sized(ClippedAdvantage):
        probe = {"batch": 3}

    assert ProbeContext().adjusted(Sized.probe).batch == 3
    assert verify_function("advantage", Sized(clip=1.0)) == []


def test_check_function_raises_e047_listing_every_finding():
    class Raises2(HuberValueLoss):
        def compute(self, batch):
            raise ValueError("x")

    with pytest.raises(ComponentContractError) as raised:
        check_function("loss", Raises2(), name="raises2")
    assert raised.value.code == "E047" and "F001" in str(raised.value) and "raises2" in str(raised.value)
    check_function("loss", HuberValueLoss(), name="ok")


def test_the_probe_is_seeded_and_reproducible():
    first = verify_function("advantage", ClippedAdvantage())
    second = verify_function("advantage", ClippedAdvantage())
    assert first == second == []
