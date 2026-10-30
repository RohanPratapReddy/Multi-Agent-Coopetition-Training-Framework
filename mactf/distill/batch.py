"""The one batch every participant sees (document 14 section 8.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..exceptions import ContractError, ValidationError

__all__ = ["DistillBatch", "coerce_batch"]


@dataclass
class DistillBatch:
    """The one batch every participant sees, plus per-peer scoring inputs.

    ``states`` is shared by construction (invariant I1). ``peer_inputs`` exists
    because a peer's module may want more than ``states`` (an attention mask, a
    target vector); a key absent for a peer falls back to ``states`` alone.
    """

    states: torch.Tensor
    peer_inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.states):
            raise ContractError(
                f"DistillBatch.states must be a tensor, got {type(self.states).__name__}"
            )
        if self.states.ndim == 0:
            raise ContractError("DistillBatch.states must have at least one dimension")
        if self.states.shape[0] < 2:
            raise ValidationError(
                f"DistillBatch batch size is {self.states.shape[0]}; contrastive "
                "distillation needs at least 2 samples for an in-batch negative"
            )

    @property
    def batch(self) -> int:
        return int(self.states.shape[0])

    @property
    def fingerprint(self) -> str:
        """Content hash of ``states``, used to bind a declared audit to a batch.

        Covers dtype, shape and the raw bytes, so two batches with equal values
        but different dtypes do not collide.
        """
        tensor = self.states.detach().to(device="cpu").contiguous()
        hasher = hashlib.sha256()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    def kwargs_for(self, peer: str) -> dict[str, Any]:
        payload = dict(self.peer_inputs.get(peer) or {})
        if "states" in payload:
            raise ContractError(
                f"peer_inputs[{peer!r}] must not override 'states'; the batch is shared"
            )
        return payload


def coerce_batch(value: Any) -> DistillBatch:
    """Accept a ``DistillBatch``, a tensor, or a mapping with a ``states`` key."""
    if isinstance(value, DistillBatch):
        return value
    if torch.is_tensor(value):
        return DistillBatch(states=value)
    if isinstance(value, Mapping):
        payload = dict(value)
        states = payload.pop("states", payload.pop("state", None))
        if states is None:
            raise ContractError(
                "a distillation batch mapping requires a 'states' key; got "
                + (", ".join(sorted(payload)) or "(nothing)")
            )
        peer_inputs = payload.pop("peer_inputs", {}) or {}
        if not isinstance(peer_inputs, Mapping):
            raise ContractError("batch 'peer_inputs' must be a mapping of peer -> mapping")
        return DistillBatch(states=states, peer_inputs=peer_inputs, extras=dict(payload))
    raise ContractError(
        f"cannot read a DistillBatch from {type(value).__name__}; pass a DistillBatch, "
        "a states tensor, or a mapping with 'states'"
    )
