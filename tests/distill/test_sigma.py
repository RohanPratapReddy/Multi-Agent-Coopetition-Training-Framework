"""Sigma reduction: modes, trust removal, bounded projection, EMA state."""

from __future__ import annotations

import math
import random

import pytest

from mactf.distill.sigma import SigmaState, sigma_weights
from mactf.exceptions import ContractError, ValidationError

LOSSES = {"a": 1.0, "b": 2.0, "c": 3.0}


def test_uniform_is_flat_without_priors_or_trust():
    out = sigma_weights(LOSSES, mode="uniform")
    assert out == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert math.fsum(out.values()) == pytest.approx(1.0, abs=1e-12)


def test_uniform_follows_priors():
    out = sigma_weights(LOSSES, mode="uniform", priors={"a": 2.0, "b": 1.0, "c": 1.0})
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.25)


def test_teacher_softmax_prefers_the_lower_loss():
    out = sigma_weights(LOSSES, mode="teacher_softmax", beta=1.0)
    assert out["a"] > out["b"] > out["c"]
    expected_a = math.exp(-1.0) / sum(math.exp(-v) for v in LOSSES.values())
    assert out["a"] == pytest.approx(expected_a, abs=1e-12)


def test_beta_zero_collapses_to_the_priors():
    out = sigma_weights(LOSSES, mode="teacher_softmax", beta=0.0)
    assert out == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_quality_softmax_reads_the_ema_not_the_loss():
    out = sigma_weights(LOSSES, mode="quality_softmax", beta=1.0, quality={"a": 0.0, "b": 0.0, "c": 5.0})
    assert out["c"] > out["a"]
    assert out["a"] == pytest.approx(out["b"])


def test_mapping_order_does_not_change_the_result():
    forward = sigma_weights(dict(LOSSES), mode="teacher_softmax")
    items = list(LOSSES.items())
    random.Random(3).shuffle(items)
    shuffled = sigma_weights(dict(items), mode="teacher_softmax")
    assert forward == shuffled  # bit-identical, not approximate


def test_large_beta_does_not_overflow():
    out = sigma_weights({"a": 0.0, "b": 1.0}, mode="teacher_softmax", beta=5000.0)
    assert out["a"] == pytest.approx(1.0)
    assert math.isfinite(out["b"])


def test_trust_removes_a_teacher_instead_of_shrinking_it():
    out = sigma_weights(LOSSES, mode="uniform", trust={"a": 0.0}, floor=0.1, cap=0.9)
    assert "a" not in out
    assert math.fsum(out.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(out.values()) >= 0.1 - 1e-12


def test_trust_at_exactly_trust_min_is_not_live():
    out = sigma_weights(LOSSES, mode="uniform", trust={"a": 0.3, "b": 1.0, "c": 1.0}, trust_min=0.3)
    assert set(out) == {"b", "c"}


def test_partial_trust_scales_the_weight():
    out = sigma_weights(LOSSES, mode="uniform", trust={"a": 0.5, "b": 1.0, "c": 1.0})
    assert out["a"] == pytest.approx(0.2)
    assert out["b"] == pytest.approx(0.4)


def test_every_teacher_untrusted_returns_empty():
    assert sigma_weights(LOSSES, mode="uniform", trust={"a": 0.0, "b": 0.0, "c": 0.0}) == {}


def test_floor_and_cap_are_respected():
    out = sigma_weights(
        {"a": 0.0, "b": 10.0, "c": 10.0}, mode="teacher_softmax", beta=3.0, floor=0.1, cap=0.7
    )
    assert math.fsum(out.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.1 - 1e-9 <= v <= 0.7 + 1e-9 for v in out.values())


def test_infeasible_box_raises_with_the_teacher_count():
    with pytest.raises(ValidationError, match="3 teachers"):
        sigma_weights(LOSSES, mode="uniform", floor=0.5, cap=1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "nope"},
        {"floor": -0.1},
        {"cap": 1.5},
        {"beta": -1.0},
        {"trust_min": 2.0},
        {"priors": {"a": 0.0}},
        {"trust": {"a": 1.5}},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValidationError):
        sigma_weights(LOSSES, **kwargs)


def test_empty_teacher_set_raises():
    with pytest.raises(ValidationError):
        sigma_weights({})


def test_non_finite_trust_and_scores_raise():
    with pytest.raises(ContractError):
        sigma_weights(LOSSES, trust={"a": float("nan")})
    with pytest.raises(ContractError):
        sigma_weights({"a": float("nan"), "b": 1.0}, mode="teacher_softmax")


def test_sigma_state_ema_matches_the_closed_form():
    state = SigmaState(decay=0.9)
    first = state.update("p", "t", 2.0)
    assert first == pytest.approx(0.1 * -2.0)
    second = state.update("p", "t", 1.0)
    assert second == pytest.approx(0.9 * first + 0.1 * -1.0)
    assert state.updates("p", "t") == 2
    assert state.get("p", "other") == 0.0


def test_sigma_state_refuses_non_finite_losses():
    state = SigmaState()
    with pytest.raises(ContractError):
        state.update("p", "t", float("inf"))


def test_sigma_state_round_trip_is_exact():
    state = SigmaState(decay=0.8)
    for loss in (1.0, 0.5, 2.5):
        state.update("p", "a", loss)
    state.update("q", "b", 0.3)
    clone = SigmaState()
    clone.load_state_dict(state.state_dict())
    assert clone.state_dict() == state.state_dict()
    assert clone.for_peer("p", ["a", "b"]) == state.for_peer("p", ["a", "b"])


def test_sigma_state_rejects_malformed_keys_and_decay():
    with pytest.raises(ValidationError):
        SigmaState(decay=1.0)
    with pytest.raises(ValidationError):
        SigmaState().load_state_dict({"values": {"nopipe": 1.0}})
