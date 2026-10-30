"""The contrastive core (document 14 section 6).

For a batch of ``B`` samples, peer ``p_i`` produces ``u_i`` of shape ``(B, d)`` and
teacher ``t_j`` produces ``w_j`` of shape ``(B, d)``, both L2-normalised. The
positive for row ``b`` of ``u_i`` is row ``b`` of ``w_j``: the same sample through
a different model.

Negatives come from three explicit groups, and the two hard-negative groups share
the row index ``b`` with the anchor and are therefore *not* in-batch negatives:

* in-batch: rows ``b' != b`` of the teacher tensor;
* cross-teacher: row ``b`` of every other teacher (passed as ``negatives``);
* cross-peer: a hinge margin (``peer_margin``), not a denominator entry, because a
  denominator entry is satisfiable by collapsing every peer onto the teacher.

Invariant I8: an inadmissible teacher row is removed from the positive term *and*
from the negative set of every other anchor. Masking the positive alone would
teach the peer to point away from that sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..exceptions import ContractError, ValidationError

__all__ = [
    "ContrastiveReport",
    "contrastive_loss",
    "l2_normalize",
    "peer_margin",
    "positive_mask",
]


def positive_mask(batch: int, *, device: torch.device) -> torch.Tensor:
    """``(B, B)`` boolean: True on the diagonal, the same-sample positions."""
    if batch < 1:
        raise ValidationError(f"positive_mask needs a batch >= 1, got {batch}")
    return torch.eye(batch, dtype=torch.bool, device=device)


def l2_normalize(tensor: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ContractError(f"l2_normalize expects a 2-D tensor, got {tuple(tensor.shape)}")
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class ContrastiveReport:
    loss: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor
    logits_mean: float
    logits_std: float
    batch: int
    temperature: float
    mask_entropy: float
    masked_rows: int = 0
    admitted_rows: int | None = None
    soft_admission: float | None = None
    clamped: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach().item()),
            "positives": float(self.positives.detach().item()),
            "negatives": float(self.negatives.detach().item()),
            "logits_mean": self.logits_mean,
            "logits_std": self.logits_std,
            "batch": float(self.batch),
            "temperature": self.temperature,
            "mask_entropy": self.mask_entropy,
            "masked_rows": float(self.masked_rows),
            "admitted_rows": float(self.admitted_rows if self.admitted_rows is not None else self.batch),
            "soft_admission": float(1.0 if self.soft_admission is None else self.soft_admission),
            "clamped": float(self.clamped),
        }


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    *,
    negatives: Sequence[torch.Tensor] = (),
    tau: float = 0.07,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
    symmetric: bool = True,
    detach_positive: bool = True,
    admission: torch.Tensor | None = None,
    negative_admission: Sequence[torch.Tensor | None] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, ContrastiveReport]:
    """Multi-negative InfoNCE between one peer and one teacher.

    ``anchor`` is the peer's ``(B, d)`` embedding and ``positive`` the teacher's
    ``(B, d)``. Each entry of ``negatives`` is another ``(B, d)`` tensor whose row
    ``b`` is a hard negative for anchor row ``b`` (a cross-teacher embedding), not
    an in-batch negative. In-batch negatives are the positive tensor's own rows
    with the diagonal removed.

    ``admission`` is the per-sample conformance mask, shape ``(B,)`` in ``[0, 1]``.
    It acts on **both** sides of the contrast:

    * on the positive logit it becomes ``log(admission)``, so a soft admission
      softens the target and a zero admission removes it;
    * on the negative logits it becomes a **column** mask, so a row that is not
      admissible is not an       anti-target either (invariant I8).

    ``negative_admission`` carries the same idea for the hard negatives: entry
    ``k`` is the ``(B,)`` admission of the teacher that produced ``negatives[k]``.
    A hard negative whose own teacher denies row ``b`` is not a negative for anchor
    ``b`` either, for the same reason.
    """
    if anchor.ndim != 2 or positive.ndim != 2:
        raise ContractError(
            "contrastive_loss expects 2-D embeddings, got "
            f"{tuple(anchor.shape)} and {tuple(positive.shape)}"
        )
    if anchor.shape != positive.shape:
        raise ContractError(
            f"contrastive_loss: anchor {tuple(anchor.shape)} and positive "
            f"{tuple(positive.shape)} must have the same shape"
        )
    batch, dim = anchor.shape
    if batch < 2:
        raise ContractError(
            "contrastive_loss needs a batch >= 2; with one sample there is no in-batch "
            "negative and the loss is identically zero"
        )
    requested = float(tau)
    if requested <= 0.0:
        raise ValidationError(f"tau must be > 0, got {tau!r}")
    if float(tau_min) <= 0.0:
        raise ValidationError(f"tau_min must be > 0, got {tau_min!r} (a zero temperature is a hard argmax)")
    if float(tau_max) < float(tau_min):
        raise ValidationError(f"tau_max {tau_max!r} must be >= tau_min {tau_min!r}")
    temperature = float(min(max(requested, float(tau_min)), float(tau_max)))

    a = l2_normalize(anchor, eps=eps)
    w = l2_normalize(positive.detach() if detach_positive else positive, eps=eps)
    hard = [l2_normalize(tensor.detach(), eps=eps) for tensor in negatives]
    for index, tensor in enumerate(hard):
        if tuple(tensor.shape) != (batch, dim):
            raise ContractError(
                f"contrastive_loss negative #{index} has shape {tuple(tensor.shape)}, "
                f"expected {(batch, dim)}"
            )

    device = a.device
    admit = _admission_column(admission, batch=batch, device=device)
    if negative_admission is not None and len(negative_admission) != len(hard):
        raise ContractError(
            f"contrastive_loss received {len(negative_admission)} negative admissions "
            f"for {len(hard)} negatives"
        )
    hard_admit = [
        _admission_column(item, batch=batch, device=device)
        for item in (negative_admission if negative_admission is not None else [None] * len(hard))
    ]
    eye = positive_mask(batch, device=device)

    positive_logit = (a * w).sum(dim=-1, keepdim=True) / temperature  # (B, 1)
    batch_logits = ((a @ w.t()) / temperature).masked_fill(eye, float("-inf"))
    hard_logits = [(a * tensor).sum(dim=-1, keepdim=True) / temperature for tensor in hard]
    logits = torch.cat([positive_logit, batch_logits, *hard_logits], dim=1)
    # hard_logits are (B, 1) per negative: one column each, because row b of a
    # hard negative is the only one that shares the anchor's sample index.

    offset = 1 + batch
    for index, item in enumerate(hard_admit):
        if item is None:
            continue
        # A hard negative from a teacher that denies row b is dead at row b only:
        # it is the single column that shares the anchor's own sample index.
        column = logits[:, offset + index].masked_fill(item <= 0, float("-inf"))
        logits = torch.cat(
            [logits[:, : offset + index], column.unsqueeze(-1), logits[:, offset + index + 1 :]],
            dim=1,
        )

    if admit is None:
        row_live = None
        masked_rows = 0
        soft_admission = None
    else:
        # Column mask first: row b of the teacher embedding is column b of the
        # in-batch block. Setting the column to -inf removes that sample from every
        # anchor's denominator (NB2: a denied sample is not a target *and* not an
        # anti-target). Each hard negative contributes the single column belonging
        # to its own row b, so it is dead exactly for the anchors whose own sample
        # is denied.
        logits = _apply_admission(logits, admit, hard=len(hard))
        row_live = torch.isfinite(logits[:, 0])
        if not bool(row_live.any()):
            raise ContractError(
                "contrastive_loss: every row is masked; a fully masked step must be "
                "skipped by the caller rather than passed here"
            )
        masked_rows = int((~row_live).sum().item())
        soft_admission = float(admit.mean().item())

    # Rows whose positive is masked contribute exactly zero and are excluded from
    # the mean. They are dropped *before* the softmax: log_softmax of a row whose
    # positive is -inf is nan, and indexing afterwards would leave it in the graph.
    live_logits = logits if row_live is None else logits[row_live]
    per_row = -torch.log_softmax(live_logits, dim=-1)[:, 0]
    loss = per_row.mean()

    with torch.no_grad():
        rows = torch.ones(batch, dtype=torch.bool, device=device) if row_live is None else row_live
        positives = (a * w).sum(dim=-1)[rows]
        positives_mean = positives.mean() if positives.numel() else torch.zeros((), device=device)
        pieces = [(a @ w.t())[~eye].reshape(-1)]
        pieces.extend((a * tensor).sum(dim=-1) for tensor in hard)
        finite = live_logits[torch.isfinite(live_logits)]
        report = ContrastiveReport(
            loss=loss.detach(),
            positives=positives_mean.detach(),
            negatives=torch.cat(pieces).mean().detach(),
            logits_mean=float(finite.mean().item()) if finite.numel() else float("nan"),
            logits_std=float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
            batch=int(batch),
            temperature=temperature,
            mask_entropy=_mask_entropy(batch),
            masked_rows=masked_rows,
            admitted_rows=batch - masked_rows,
            soft_admission=soft_admission,
            clamped=abs(temperature - requested) > 1e-12,
        )

    if symmetric:
        reverse, _ = contrastive_loss(
            positive,
            anchor,
            negatives=negatives,
            tau=temperature,
            tau_min=temperature,
            tau_max=temperature,
            symmetric=False,
            detach_positive=False,
            admission=admission,
            negative_admission=negative_admission,
            eps=eps,
        )
        loss = 0.5 * (loss + reverse)
        report.loss = loss.detach()
    return loss, report


def _admission_column(
    admission: torch.Tensor | None, *, batch: int, device: torch.device
) -> torch.Tensor | None:
    """Validate an admission tensor and move it to ``device`` as float32."""
    if admission is None:
        return None
    if not isinstance(admission, torch.Tensor):
        raise ContractError(
            f"contrastive_loss.admission must be a tensor, got {type(admission).__name__}"
        )
    if tuple(admission.shape) != (batch,):
        raise ContractError(
            f"contrastive_loss.admission must have shape ({batch},), got {tuple(admission.shape)}"
        )
    # Admission is a property of a teacher's output measured under no_grad. A
    # gradient through it would let a peer negotiate its own admission.
    admit = admission.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(admit).all()):
        raise ContractError(
            "contrastive_loss.admission contains a non-finite value; NaN must be "
            "resolved to 0 (denied) in conformance.admission, not here"
        )
    if bool((admit < 0).any()) or bool((admit > 1).any()):
        raise ContractError(
            "contrastive_loss.admission must lie in [0, 1]; clamp it in "
            "conformance.admission, not here"
        )
    return admit


def _apply_admission(logits: torch.Tensor, admit: torch.Tensor, *, hard: int) -> torch.Tensor:
    """Apply an admission vector to ``[positive | in-batch (B) | hard (1 each)]``.

    * In-batch block: column ``b`` is teacher sample ``b``; it is killed for every
      anchor when ``admit[b] == 0`` (I8).
    * Hard block ``k``: its single column belongs to the anchor's own row, so it
      is killed only on the rows where ``admit[b] == 0``. Those rows are dropped
      anyway, so this keeps the masked softmax exact without touching live rows.
    * Positive column: scaled by ``log(admit)``, so a soft admission keeps a
      gradient path and a zero admission becomes ``-inf``.
    """
    batch = int(admit.shape[0])
    live = admit > 0
    in_batch = logits[:, 1 : 1 + batch].masked_fill(~live.unsqueeze(0), float("-inf"))
    positive = logits[:, :1] + _admission_log(admit).unsqueeze(-1).to(dtype=logits.dtype)
    pieces = [positive, in_batch]
    if hard:
        tail = logits[:, 1 + batch :].masked_fill(~live.unsqueeze(-1), float("-inf"))
        pieces.append(tail)
    return torch.cat(pieces, dim=1)


def _admission_log(admit: torch.Tensor) -> torch.Tensor:
    """``log(admission)`` with an exact ``-inf`` at zero."""
    safe = admit.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(admit > 0, torch.log(safe), torch.full_like(safe, float("-inf")))


def _mask_entropy(batch: int) -> float:
    """``log(B - 1)``: the entropy of the uniform distribution over in-batch negatives.

    Reported so a degenerate batch (or a silently shrunk one) is visible in the log
    line rather than inferred from a suspiciously small loss.
    """
    if batch <= 1:
        return 0.0
    return float(math.log(batch - 1))


def peer_margin(
    anchor: torch.Tensor,
    peers: Mapping[str, torch.Tensor],
    *,
    margin: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalise an anchor pointing the same way as a sibling peer.

    ``peers`` maps sibling path -> that sibling's ``(B, d)`` embedding for the same
    batch. The penalty is ``relu(cos(anchor, sibling) - margin)``, averaged over
    siblings and rows. Being a hinge, a peer already far from its siblings
    contributes exactly zero and receives exactly zero gradient.

    Siblings are detached: peer ``p_i``'s objective may depend on where a sibling
    *is*, but must not push it, or ``L_i`` would move ``p_j``'s parameters and
    invariant I3 would be false.
    """
    if not peers:
        return torch.zeros((), device=anchor.device, dtype=anchor.dtype)
    if not 0.0 <= float(margin) <= 1.0:
        raise ValidationError(f"peer margin must be in [0, 1], got {margin!r}")
    a = l2_normalize(anchor, eps=eps)
    total = torch.zeros((), device=a.device, dtype=a.dtype)
    for name in sorted(peers):
        other = l2_normalize(peers[name].detach(), eps=eps)
        if other.shape != a.shape:
            raise ContractError(
                f"peer_margin: sibling {name!r} has shape {tuple(other.shape)}, "
                f"anchor has {tuple(a.shape)}"
            )
        total = total + torch.relu((a * other).sum(dim=-1) - float(margin)).mean()
    return total / len(peers)
