"""The cooperative pool: gating, capping, proportional scaling, order invariance."""

from __future__ import annotations

import random

import pytest

from mactf.exceptions import ContractError, ImproperlyConfigured
from mactf.coopetition.pool import (
    COOPERATIVE_RULES,
    CooperativeAllocation,
    CooperativeBudget,
    CooperativePool,
    cooperative_total,
)


def test_cooperative_rules_are_r10_r11_r16():
    assert COOPERATIVE_RULES == frozenset({"R10", "R11", "R16"})


def test_requests_within_budget_are_granted_in_full():
    pool = CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=3.0))
    allocation = pool.allocate({"a": 1.0, "b": 2.0, "c": 0.5})
    assert allocation.awards == {"a": 1.0, "b": 2.0, "c": 0.5}
    assert allocation.granted == pytest.approx(3.5)
    assert allocation.denied == pytest.approx(0.0)
    assert allocation.reason is None


def test_exactly_fitting_requests_are_granted_in_full():
    pool = CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=6.0))
    allocation = pool.allocate({"a": 3.0, "b": 3.0})
    assert allocation.granted == pytest.approx(6.0)
    assert allocation.denied == pytest.approx(0.0)
    assert allocation.reason is None


def test_over_budget_requests_are_scaled_proportionally():
    pool = CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=3.0))
    allocation = pool.allocate({"a": 3.0, "b": 3.0, "c": 3.0})
    # 9 requested, 6 available -> scale 2/3.
    assert allocation.awards["a"] == pytest.approx(2.0)
    assert allocation.awards["b"] == pytest.approx(2.0)
    assert allocation.awards["c"] == pytest.approx(2.0)
    assert allocation.granted == pytest.approx(6.0)
    assert allocation.denied == pytest.approx(3.0)
    assert "scaled by" in (allocation.reason or "")


def test_denied_sums_correctly_across_peers():
    pool = CooperativePool(CooperativeBudget(per_episode=4.0, per_peer_cap=4.0))
    allocation = pool.allocate({"a": 2.0, "b": 4.0})
    assert allocation.granted == pytest.approx(4.0)
    assert allocation.denied == pytest.approx(2.0)
    # The withheld amount is attributable per peer, not just in aggregate.
    assert allocation.denied_for("a") == pytest.approx(2.0 / 3.0)
    assert allocation.denied_for("b") == pytest.approx(4.0 / 3.0)
    assert allocation.denied_for("a") + allocation.denied_for("b") == pytest.approx(
        allocation.denied
    )


def test_per_peer_cap_clamps_before_the_total_is_considered():
    pool = CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=1.0))
    allocation = pool.allocate({"a": 5.0, "b": 5.0})
    # Each is capped at 1.0, so the total (2.0) now fits and nothing is scaled.
    assert allocation.awards == {"a": 1.0, "b": 1.0}
    assert allocation.granted == pytest.approx(2.0)
    assert allocation.denied == pytest.approx(8.0)
    assert allocation.reason is None


def test_ineligible_peers_get_zero_and_do_not_consume_budget():
    pool = CooperativePool(CooperativeBudget(per_episode=3.0, per_peer_cap=3.0))
    allocation = pool.allocate(
        {"a": 3.0, "b": 3.0},
        eligible={"a": True, "b": False},
    )
    assert allocation.awards["a"] == pytest.approx(3.0)
    assert allocation.awards["b"] == pytest.approx(0.0)
    assert allocation.granted == pytest.approx(3.0)
    assert allocation.reason is None
    # A void peer's request is still reported, and it is fully denied.
    assert allocation.requested == {"a": 3.0, "b": 3.0}
    assert allocation.denied_for("b") == pytest.approx(3.0)


def test_missing_eligibility_entry_defaults_to_eligible():
    pool = CooperativePool(CooperativeBudget(per_episode=6.0))
    assert pool.allocate({"a": 1.0}, eligible={}).awarded_to("a") == pytest.approx(1.0)


def test_allocation_is_invariant_to_input_ordering():
    budget = CooperativeBudget(per_episode=5.0, per_peer_cap=2.5)
    requests = {f"p{i}": 1.0 + 0.1 * i for i in range(8)}
    baseline = CooperativePool(budget).allocate(requests)

    items = list(requests.items())
    for seed in range(10):
        random.Random(seed).shuffle(items)
        shuffled = dict(items)
        candidate = CooperativePool(budget).allocate(shuffled)
        assert candidate.awards == baseline.awards
        assert candidate.granted == baseline.granted
        assert candidate.denied == baseline.denied


def test_zero_budget_grants_nothing():
    pool = CooperativePool(CooperativeBudget(per_episode=0.0, per_peer_cap=0.0))
    allocation = pool.allocate({"a": 1.0})
    assert allocation.awards == {"a": 0.0}
    assert allocation.granted == pytest.approx(0.0)
    assert allocation.denied == pytest.approx(1.0)


def test_empty_requests_yield_an_empty_allocation():
    allocation = CooperativePool().allocate({})
    assert allocation.awards == {}
    assert allocation.granted == pytest.approx(0.0)
    assert allocation.denied == pytest.approx(0.0)


def test_carry_over_rolls_unspent_budget_forward():
    pool = CooperativePool(
        CooperativeBudget(per_episode=4.0, per_peer_cap=6.0, carry_over=True)
    )
    pool.allocate({"a": 1.0})
    assert pool.carry == pytest.approx(3.0)
    assert pool.available == pytest.approx(7.0)

    second = pool.allocate({"a": 6.0})
    assert second.denied == pytest.approx(0.0)
    assert second.granted == pytest.approx(6.0)
    assert pool.carry == pytest.approx(1.0)


def test_carry_over_does_not_accumulate_when_disabled():
    pool = CooperativePool(
        CooperativeBudget(per_episode=4.0, per_peer_cap=4.0, carry_over=False)
    )
    pool.allocate({"a": 1.0})
    assert pool.carry == pytest.approx(0.0)
    assert pool.available == pytest.approx(4.0)


def test_negative_requests_are_rejected():
    with pytest.raises(ContractError):
        CooperativePool().allocate({"a": -1.0})


def test_non_finite_requests_are_rejected():
    with pytest.raises(ContractError):
        CooperativePool().allocate({"a": float("nan")})
    with pytest.raises(ContractError):
        CooperativePool().allocate({"a": float("inf")})


def test_budget_rejects_negative_values():
    with pytest.raises(ImproperlyConfigured):
        CooperativeBudget(per_episode=-1.0)
    with pytest.raises(ImproperlyConfigured):
        CooperativeBudget(per_peer_cap=float("nan"))


def test_budget_from_settings_tolerates_partial_mappings():
    budget = CooperativeBudget.from_settings({"PER_EPISODE": 2.0})
    assert budget.per_episode == 2.0
    assert budget.per_peer_cap == 3.0
    assert budget.carry_over is False
    assert CooperativeBudget.from_settings(None) == CooperativeBudget()


def test_allocation_as_dict_is_json_shaped():
    allocation = CooperativePool().allocate({"a": 1.0})
    payload = allocation.as_dict()
    assert payload["awards"] == {"a": 1.0}
    assert payload["granted"] == pytest.approx(1.0)
    assert payload["requested"] == {"a": 1.0}
    assert payload["reason"] is None
    assert isinstance(allocation, CooperativeAllocation)


def test_cooperative_total_reads_per_rule_accounting():
    class Reward:
        by_rule = {"R1": 10.0, "R10": 2.0, "R11": 1.5, "R16": 1.5}

    assert cooperative_total(Reward()) == pytest.approx(5.0)


def test_cooperative_total_is_zero_without_per_rule_accounting():
    class Reward:
        pass

    assert cooperative_total(Reward()) == pytest.approx(0.0)
