"""The contrastive core: the section 6.7 worked examples and the gradient gates."""

from __future__ import annotations

import math

import pytest
import torch

from mactf.distill.contrastive import (
    contrastive_loss,
    l2_normalize,
    peer_margin,
    positive_mask,
)
from mactf.exceptions import ContractError, ValidationError

U = l2_normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=torch.float64))
W = l2_normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=torch.float64))


def _loss(anchor=U, positive=W, **kwargs):
    kwargs.setdefault("tau", 0.5)
    kwargs.setdefault("symmetric", False)
    return contrastive_loss(anchor, positive, **kwargs)


def test_worked_example_matches_hand_computation():
    loss, report = _loss()
    assert float(loss) == pytest.approx(0.615200, abs=1e-5)
    assert report.batch == 3
    assert report.masked_rows == 0
    assert report.temperature == pytest.approx(0.5)


def test_masked_row_removes_the_column_too():
    admission = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    loss, report = _loss(admission=admission)
    assert float(loss) == pytest.approx(0.454805, abs=1e-5)
    # The positive-only mask would give 0.692614; removal must not.
    assert abs(float(loss) - 0.692614) > 1e-3
    assert report.masked_rows == 1
    assert report.admitted_rows == 2


def test_soft_admission_raises_the_term():
    admission = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64)
    loss, report = _loss(admission=admission)
    assert float(loss) == pytest.approx(0.719882, abs=1e-5)
    assert report.masked_rows == 0


def test_masked_loss_is_lower_than_unmasked_with_a_near_duplicate():
    base = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=torch.float64)
    anchor = l2_normalize(base)
    positive = l2_normalize(base + 0.001)
    admission = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    masked, _ = _loss(anchor, positive, admission=admission)
    unmasked, _ = _loss(anchor, positive)
    assert float(masked) < float(unmasked)

    # Removing sample 1 as a column and a row is exactly the loss of the batch without it.
    keep = [0, 2]
    reference, _ = _loss(anchor[keep], positive[keep])
    assert float(masked) == pytest.approx(float(reference), abs=1e-9)


def test_identical_embeddings_give_the_minimum():
    eye = torch.eye(4, dtype=torch.float64)
    loss, _ = contrastive_loss(eye, eye, tau=0.07, symmetric=False)
    assert float(loss) < 1e-3


def test_orthogonal_embeddings_give_log_batch():
    batch = 8
    positive = torch.eye(batch + 1, dtype=torch.float64)[:batch]
    anchor = torch.eye(batch + 1, dtype=torch.float64)[batch:].repeat(batch, 1)
    loss, _ = contrastive_loss(anchor, positive, tau=1.0, tau_max=1.0, symmetric=False)
    assert float(loss) == pytest.approx(math.log(batch), abs=1e-3)


def test_duplicate_rows_stay_finite():
    row = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    loss, report = contrastive_loss(row, row.clone(), tau=0.1, symmetric=True)
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in report.as_dict().values())


def test_forced_duplicate_positive_with_denial_has_no_nan():
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    positive = anchor.detach().clone()
    admission = torch.tensor([0.0, 1.0], dtype=torch.float64)
    loss, _ = contrastive_loss(anchor, positive, tau=0.1, admission=admission)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(anchor.grad).all()


def test_symmetric_loss_is_symmetric():
    forward, _ = contrastive_loss(U, W, tau=0.5, symmetric=True)
    swapped, _ = contrastive_loss(W, U, tau=0.5, symmetric=True)
    assert float(forward) == pytest.approx(float(swapped), abs=1e-9)


def test_tau_clamp_is_reported():
    _, report = contrastive_loss(U, W, tau=1e-6, tau_min=0.01, symmetric=False)
    assert report.clamped is True
    assert report.temperature == pytest.approx(0.01)


@pytest.mark.parametrize("tau", [0.0, -0.1])
def test_non_positive_tau_raises(tau):
    with pytest.raises((ValidationError, ContractError)):
        contrastive_loss(U, W, tau=tau)


def test_batch_of_one_is_refused_naming_the_negative_requirement():
    with pytest.raises(ContractError, match="batch >= 2"):
        contrastive_loss(U[:1], W[:1], tau=0.5)


def test_non_two_dimensional_is_refused():
    with pytest.raises(ContractError):
        contrastive_loss(U.reshape(3, 1, 2), W.reshape(3, 1, 2), tau=0.5)


def test_shape_mismatch_is_refused():
    with pytest.raises(ContractError):
        contrastive_loss(U, W[:, :1], tau=0.5)


def test_positive_is_detached():
    anchor = U.clone().requires_grad_(True)
    positive = W.clone().requires_grad_(True)
    loss, _ = contrastive_loss(anchor, positive, tau=0.5, symmetric=False, detach_positive=True)
    loss.backward()
    assert anchor.grad is not None
    assert positive.grad is None


def test_symmetric_direction_keeps_the_anchor_gradient():
    anchor = U.clone().requires_grad_(True)
    positive = W.clone().requires_grad_(True)
    loss, _ = contrastive_loss(anchor, positive, tau=0.5, symmetric=True, detach_positive=True)
    loss.backward()
    assert anchor.grad is not None
    # The reverse direction makes the positive tensor an anchor (doc 14 section 6.3 rule 1).
    assert positive.grad is not None


def test_report_fields_are_finite():
    _, report = _loss()
    for value in report.as_dict().values():
        assert math.isfinite(value)


def test_positive_mask_is_the_diagonal():
    mask = positive_mask(3, device=torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.eye(3, dtype=torch.bool))
    with pytest.raises(ValidationError):
        positive_mask(0, device=torch.device("cpu"))


def test_hard_negative_from_another_teacher_raises_the_loss():
    other = l2_normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]], dtype=torch.float64))
    plain, _ = _loss()
    harder, _ = _loss(negatives=[other])
    assert float(harder) > float(plain)


def test_hard_negative_whose_teacher_denies_the_row_is_removed():
    other = l2_normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]], dtype=torch.float64))
    denied = torch.zeros(3, dtype=torch.float64)
    with_denied, _ = _loss(negatives=[other], negative_admission=[denied])
    plain, _ = _loss()
    assert float(with_denied) == pytest.approx(float(plain), abs=1e-9)


def test_peer_margin_is_zero_when_peers_are_far_apart_and_has_no_gradient():
    own = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    sibling = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    value = peer_margin(own, {"sib": sibling}, margin=0.25)
    assert float(value) == 0.0
    value.backward()
    assert own.grad is None or float(own.grad.abs().sum()) == 0.0


def test_peer_margin_detaches_siblings():
    own = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    sibling = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    value = peer_margin(own, {"sib": sibling}, margin=0.25)
    assert float(value) > 0.0
    value.backward()
    assert own.grad is not None
    assert sibling.grad is None
