"""verify_weight_update: a strategy proves itself on five synthetic groups before it runs (document 17 section 10.6)."""

from __future__ import annotations

import pytest

from mactf.coopetition.weight_probe import verify_weight_update
from mactf.coopetition.weights import (
    KappaSoftmaxAverage,
    NoAggregation,
    ReferenceOnlyAverage,
    ScoreWeightedAverage,
    UniformAverage,
    WeightStrategy,
    WeightTraits,
    make_report,
)
from mactf.exceptions import ComplianceError, ImproperlyConfigured

KW = {"floor": 0.02, "cap": 0.6, "tolerance": 1e-6, "slot": "TRAINER.WEIGHT_UPDATE"}


def probe(factory, **overrides):
    verify_weight_update(factory, **{**KW, **overrides})


class Softmax(WeightStrategy):
    """A plain softmax: it knows nothing of the floor and cap."""

    name = "softmax"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        import math

        names = sorted(vectors)
        active = [n for n in names if (eligible or {}).get(n, True)]
        if not active:
            return make_report({}, names, kappa=kappa, eligible=eligible, fallback=True)
        top = max(scores.get(n, 0.0) for n in active)
        exps = {n: math.exp(kappa * 3.0 * (scores.get(n, 0.0) - top)) for n in active}
        total = sum(exps.values())
        return make_report({n: e / total for n, e in exps.items()}, names, kappa=kappa, eligible=eligible)


class Fixed(WeightStrategy):
    """Always returns ``weights`` for the names it is given (padded with zeros)."""

    name = "fixed"

    def __init__(self, weights=None, error=None, returns=None):
        self.weights = weights
        self.error = error
        self.returns = returns

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        if self.error is not None:
            raise self.error
        if self.returns is not None:
            return self.returns
        names = sorted(vectors)
        return make_report(self.weights(names) if callable(self.weights) else self.weights, names,
                           kappa=kappa, eligible=eligible)


def test_the_builtins_pass_their_own_probe() -> None:
    for factory in (lambda: UniformAverage(), lambda: ScoreWeightedAverage(), lambda: ReferenceOnlyAverage(),
                    lambda: NoAggregation()):
        probe(factory)


def test_a_plain_softmax_passes_under_repair_because_the_guard_projects_the_bounds() -> None:
    probe(lambda: Softmax())                                  # the document's ema_softmax is of this kind
    with pytest.raises(ImproperlyConfigured, match="E064.*U005.*outside"):
        probe(lambda: Softmax(), mode="fallback")             # under fallback/enforce the same output is replaced every epoch
    with pytest.raises(ImproperlyConfigured, match="E064.*U005"):
        probe(lambda: Softmax(), mode="enforce")


def test_a_vector_that_does_not_sum_to_one_fails_in_every_mode() -> None:
    three = lambda: Fixed(lambda names: {n: 3.0 / len(names) for n in names})        # noqa: E731
    for mode in ("repair", "fallback", "enforce"):
        with pytest.raises(ImproperlyConfigured, match="E064.*weights sum to 3"):
            probe(three, mode=mode)


@pytest.mark.parametrize("weights,match", [
    (lambda names: {n: float("nan") for n in names}, "U001"),
    (lambda names: {names[0]: -1.0, names[1]: 2.0}, "U002"),
    (lambda names: {n: 1.0 / len(names) for n in names}, "U004"),           # void peers carry weight
])
def test_hard_and_void_violations_fail_in_every_mode(weights, match) -> None:
    for mode in ("repair", "fallback"):
        with pytest.raises(ImproperlyConfigured, match=f"E064.*{match}"):
            probe(lambda: Fixed(weights), mode=mode)


def test_findings_name_the_case_that_found_them() -> None:
    with pytest.raises(ImproperlyConfigured) as caught:
        probe(lambda: Fixed(lambda names: {n: 1.0 / len(names) for n in names}))
    text = str(caught.value)
    assert "one_void:" in text and "two_void:" in text


def test_apply_raising_is_a_finding_but_a_compliance_refusal_is_allowed() -> None:
    with pytest.raises(ImproperlyConfigured, match="E064.*U007.*KeyError.*boom"):
        probe(lambda: Fixed(error=KeyError("boom")))
    probe(lambda: Fixed(error=ComplianceError("cannot serve this case")))     # refusing is allowed


def test_a_wrong_return_type_is_a_finding() -> None:
    with pytest.raises(ImproperlyConfigured, match="E064.*returned dict, not an AggregationReport"):
        probe(lambda: Fixed(returns={"a": 1.0}))


def test_a_strategy_that_writes_nothing_is_skipped() -> None:
    class Quiet(Fixed):
        traits = WeightTraits(aggregates=False)

    probe(lambda: Quiet(error=RuntimeError("never called")))


def test_an_infeasible_case_is_skipped_not_blamed() -> None:
    """With a floor of 0.3, four participants cannot sum to 1: the aggregator itself refuses that group."""
    class BreaksOnFour(Softmax):
        def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
            if sum(1 for n in vectors if (eligible or {}).get(n, True)) == 4:
                raise KeyError("four peers")
            return super().apply(vectors, scores, kappa=kappa, level=level, eligible=eligible)

    with pytest.raises(ImproperlyConfigured, match="E064.*U007.*four peers"):
        probe(lambda: BreaksOnFour())
    probe(lambda: BreaksOnFour(), floor=0.3, cap=0.6)          # the four-peer cases are skipped, the rest pass


def test_each_case_gets_a_fresh_instance() -> None:
    made = []

    def factory():
        made.append(object())
        return Softmax()

    probe(factory)
    assert len(made) >= 5                                      # five cases, plus the state check


# -- state (E067) ---------------------------------------------------------------------------------------------
class Stateful(Softmax):
    def __init__(self):
        self.calls = 0

    def apply(self, *args, **kwargs):
        self.calls += 1
        return super().apply(*args, **kwargs)

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, payload):
        self.calls = int(payload["calls"])


def test_a_stateful_strategy_that_round_trips_passes() -> None:
    probe(lambda: Stateful())


def test_e067_state_must_be_json() -> None:
    class NotJson(Stateful):
        def state_dict(self):
            return {"x": {1, 2}}

    class Nan(Stateful):
        def state_dict(self):
            return {"x": float("nan")}

    for cls in (NotJson, Nan):
        with pytest.raises(ImproperlyConfigured, match="E067.*not JSON-serialisable"):
            probe(lambda cls=cls: cls())


def test_e067_state_needs_a_load_and_must_round_trip() -> None:
    class NoLoad(Softmax):
        def state_dict(self):
            return {"n": 1}

    with pytest.raises(ImproperlyConfigured, match="E067.*no load_state_dict"):
        probe(lambda: NoLoad())

    class Drifts(Stateful):
        def load_state_dict(self, payload):
            self.calls = int(payload["calls"]) + 1

    with pytest.raises(ImproperlyConfigured, match="E067.*changed across a load_state_dict"):
        probe(lambda: Drifts())

    class Rejects(Stateful):
        def load_state_dict(self, payload):
            raise ValueError("not mine")

    with pytest.raises(ImproperlyConfigured, match="E067.*rejected its own state_dict"):
        probe(lambda: Rejects())


def test_the_guarded_default_with_a_subclassed_aggregator_passes_its_probe() -> None:
    from mactf.coopetition.aggregation import Aggregator
    from mactf.coopetition.temperature import AdaptiveKappa

    class Sub(Aggregator):
        pass

    probe(lambda: KappaSoftmaxAverage(Sub(controller=AdaptiveKappa(num_peers=4))))
