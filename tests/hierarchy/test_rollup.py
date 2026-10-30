"""Rollup strategies (document 10 section 5.2).

Every strategy except ``ExplicitRollup`` unions the children's ``void_rules``. A
``MeanRollup`` across a voided child and two healthy children produces a number;
that number must not remove the void.
"""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ContractError, ValidationError
from mactf.rl.functions.rollup import (
    ExplicitRollup,
    KappaWeightedRollup,
    MeanRollup,
    MinimumRollup,
    SumRollup,
    WorstCaseRollup,
)
from mactf.scoring.base import ScorerOutput, ScoringKind
from mactf.scoring.hierarchy import ROLLUP_STRATEGIES, build_rollup


def _child(amount: float, *, rules=(), void=()):
    return ScorerOutput(
        kind=ScoringKind.REWARD,
        amount=amount,
        rules=tuple(rules),
        void_rules=tuple(void),
    )


CHILDREN = [
    _child(1.0, rules=("R1",)),
    _child(2.0, rules=("R5",), void=("P1",)),
    _child(3.0, rules=("R6",)),
]


def test_mean_rollup_averages_the_amounts():
    result = MeanRollup().rollup("peer-1", CHILDREN)
    assert result.amount == pytest.approx(2.0)
    assert result.kind is ScoringKind.HIERARCHY


def test_sum_rollup_adds_the_amounts():
    assert SumRollup().rollup("peer-1", CHILDREN).amount == pytest.approx(6.0)


def test_minimum_rollup_takes_the_worst_child():
    result = MinimumRollup().rollup("peer-1", CHILDREN)
    assert result.amount == pytest.approx(1.0)
    assert result.breakdown["worst_index"] == 0


def test_worst_case_rollup_is_numerically_the_minimum():
    assert WorstCaseRollup().rollup("peer-1", CHILDREN).amount == pytest.approx(1.0)


def test_kappa_rollup_normalises_weights():
    weights = {"R1": 0.2, "R5": 0.3, "R6": 0.5}
    result = KappaWeightedRollup().rollup("peer-1", CHILDREN, weights=weights)
    expected = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0
    assert result.amount == pytest.approx(expected)
    assert sum(result.breakdown["weights"]) == pytest.approx(1.0)


def test_kappa_rollup_falls_back_to_the_mean_when_weights_sum_to_zero():
    zero = {"R1": 0.0, "R5": 0.0, "R6": 0.0}
    result = KappaWeightedRollup().rollup("peer-1", CHILDREN, weights=zero)
    assert result.amount == pytest.approx(2.0)
    assert result.breakdown["fallback_used"] is True


def test_kappa_rollup_can_raise_on_zero_weights():
    zero = {"R1": 0.0, "R5": 0.0, "R6": 0.0}
    with pytest.raises(ContractError, match="sum to zero"):
        KappaWeightedRollup(fallback="error").rollup("peer-1", CHILDREN, weights=zero)


def test_kappa_rollup_rejects_an_unknown_fallback():
    with pytest.raises(ValidationError, match="unknown fallback"):
        KappaWeightedRollup(fallback="explode")


@pytest.mark.parametrize(
    "strategy",
    [MeanRollup(), SumRollup(), MinimumRollup(), WorstCaseRollup(), KappaWeightedRollup()],
)
def test_every_non_explicit_strategy_unions_the_voids(strategy):
    result = strategy.rollup("peer-1", CHILDREN)
    assert result.void_rules == ("P1",)


def test_explicit_rollup_owns_the_whole_output_including_the_voids():
    def fn(parent, children, *, weights, scores):
        return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=0.0)

    result = ExplicitRollup(fn).rollup("peer-1", CHILDREN)
    assert result.void_rules == ()


def test_explicit_rollup_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        ExplicitRollup()


def test_explicit_rollup_rejects_a_non_output_result():
    with pytest.raises(ContractError, match="not ScorerOutput"):
        ExplicitRollup(lambda *args, **kwargs: 3.0).rollup("peer-1", CHILDREN)


def test_an_empty_child_list_is_zero_not_an_error():
    for strategy in (MeanRollup(), MinimumRollup(), SumRollup(), KappaWeightedRollup()):
        assert strategy.rollup("peer-1", []).amount == 0.0


def test_rollup_rules_are_the_union_of_child_rules():
    result = MeanRollup().rollup("peer-1", CHILDREN)
    assert result.rules == ("R1", "R5", "R6")


def test_breakdown_reports_the_child_amounts_and_strategy():
    payload = KappaWeightedRollup().rollup("peer-1", CHILDREN).breakdown
    assert payload["strategy"] == "kappa"
    assert payload["child_amounts"] == [1.0, 2.0, 3.0]
    assert payload["children"] == 3
    assert payload["parent"] == "peer-1"


def test_registry_builds_every_strategy():
    for name in ROLLUP_STRATEGIES:
        if name == "explicit":
            # ``explicit`` cannot have a default: the function owns the output.
            built = build_rollup(
                {"type": "explicit", "fn": lambda *args, **kwargs: _child(0.0)}
            )
        else:
            built = build_rollup(name)
        assert isinstance(built, ROLLUP_STRATEGIES[name])


def test_an_explicit_strategy_without_a_function_names_the_requirement():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        build_rollup("explicit")


def test_registry_rejects_an_unknown_strategy():
    with pytest.raises(ValidationError, match="unknown rollup"):
        build_rollup("median")


def test_registry_accepts_a_spec_mapping():
    built = build_rollup({"type": "kappa", "fallback": "zero"})
    assert built.fallback == "zero"


def test_build_rollup_passes_a_strategy_through_unchanged():
    strategy = MeanRollup()
    assert build_rollup(strategy) is strategy
    assert build_rollup(None).name == "kappa"


def test_from_spec_rejects_a_mismatched_type():
    with pytest.raises(ValidationError, match="cannot build type"):
        MeanRollup.from_spec({"type": "sum"})


def test_describe_is_serializable():
    payload = KappaWeightedRollup().describe()
    assert payload["type"] == "kappa"
    assert payload["normalise"] is True
    assert payload["fallback"] == "mean"


def test_the_rollup_protocol_is_satisfied():
    from mactf.rl.functions.rollup import RollupStrategy

    assert isinstance(MeanRollup(), RollupStrategy)
    assert isinstance(KappaWeightedRollup(), RollupStrategy)


def test_tensors_are_not_required_by_the_contract():
    """Rollups deal in floats; a caller with tensors converts, as the scorer does."""
    result = MeanRollup().rollup("peer-1", [_child(0.5), _child(1.5)])
    assert torch.is_tensor(torch.tensor(result.amount))
    assert result.amount == pytest.approx(1.0)
