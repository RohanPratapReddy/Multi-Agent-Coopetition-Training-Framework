"""The trainer protocol: value objects and the structural contract (document 08 section 3).

``StepResult`` and ``EpochResult`` are what a custom ``trainer`` returns and what a
callback receives. ``TrainerProtocol`` is structural, so a user class needs no
base class and no import beyond ``StepResult`` if it wants typed results.

``PeerNode | Peer`` appears in the signatures deliberately: a depth-1 tree always
gives a ``PeerNode``, but a single-module implementation should not have to import
tree types. ``node.peer`` is the ``Peer`` in both cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

import torch

from ..rl.loss import LossBreakdown

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..coopetition.peer import Peer
    from ..coopetition.tree import PeerNode, PeerTree

__all__ = [
    "EpochResult",
    "TrainResult",
    "TrainerCapabilities",
    "TrainerProtocol",
    "StepResult",
    "coerce_step_result",
]


@dataclass(frozen=True)
class StepResult:
    """One training step's outcome.

    ``stepped`` is false when gradient accumulation absorbed the step, so a
    callback can tell "the optimizer moved" from "the loss was measured".
    """

    loss: torch.Tensor
    breakdown: LossBreakdown = field(default_factory=LossBreakdown)
    stepped: bool = True
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def loss_value(self) -> float:
        """The scalar loss, detached, for logging."""
        return float(self.loss.detach().item())

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss_value,
            "breakdown": self.breakdown.as_dict(),
            "stepped": self.stepped,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class EpochResult:
    """One epoch's outcome, passed to callbacks and the logger."""

    epoch: int
    steps: int
    mean_loss: float
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    #: True only when the epoch completed its full step budget. A truncated epoch
    #: (cancellation, a watchdog pause) is reported rather than hidden.
    completed: bool = True
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "steps": self.steps,
            "mean_loss": self.mean_loss,
            "metrics": {key: dict(value) for key, value in self.metrics.items()},
            "completed": self.completed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrainerCapabilities:
    """What this trainer can actually do, so the service layer can say so."""

    optimizer_granularity: str = "step"
    supports_amp: bool = False
    supports_hierarchy: bool = False
    supports_peft: bool = False
    supports_aggregation: bool = True
    deterministic: bool = True
    supports_distillation: bool = False
    distillation_conformance: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_granularity": self.optimizer_granularity,
            "supports_amp": self.supports_amp,
            "supports_hierarchy": self.supports_hierarchy,
            "supports_peft": self.supports_peft,
            "supports_aggregation": self.supports_aggregation,
            "deterministic": self.deterministic,
            "supports_distillation": self.supports_distillation,
            "distillation_conformance": self.distillation_conformance,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrainResult:
    """The outcome of ``MACTFTrainer.fit``."""

    epochs: int
    steps: int
    mean_loss: float
    history: tuple[EpochResult, ...] = ()
    status: str = "completed"
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    #: The episode scorecards the trainer produced, oldest first. Not part of ``as_dict``:
    #: scorecards have their own wire form and can be large.
    episodes: tuple[Any, ...] = ()
    #: Notes from the component verification of a resume (document 17 section 14.5).
    component_notes: tuple[str, ...] = ()

    @property
    def halted(self) -> bool:
        return self.status != "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "mean_loss": self.mean_loss,
            "history": [result.as_dict() for result in self.history],
            "status": self.status,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            **({"component_notes": list(self.component_notes)} if self.component_notes else {}),
        }


@runtime_checkable
class TrainerProtocol(Protocol):
    """What :class:`~mactf.training.trainer.MACTFTrainer` accepts as ``trainer=``."""

    def setup(self, node: "PeerNode | Peer") -> None: ...

    def train_step(
        self, node: "PeerNode | Peer", batch: Mapping[str, Any]
    ) -> StepResult: ...

    def on_epoch_end(
        self, tree: "PeerTree | Mapping[str, Peer]", result: EpochResult
    ) -> None: ...


def coerce_step_result(value: Any, *, node: Any = None) -> StepResult:
    """Normalize whatever a custom step returned into a :class:`StepResult`.

    Accepted: ``StepResult``, ``torch.Tensor``, ``(loss, dict)``, or a mapping with
    a ``loss`` key. Anything else is a contract error rather than a silent
    ``float()`` coercion, because a custom step returning the wrong object is a
    bug the user needs to see.
    """
    from ..exceptions import ContractError

    if isinstance(value, StepResult):
        return value
    if torch.is_tensor(value):
        return StepResult(loss=value)
    if isinstance(value, (tuple, list)):
        if not value:
            raise ContractError("a custom train_step returned an empty tuple")
        loss = value[0]
        extras = dict(value[1]) if len(value) > 1 and value[1] else {}
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(float(loss), dtype=torch.float32, requires_grad=True)
        breakdown = LossBreakdown(
            total=float(loss.detach().item()),
            extras={str(key): float(val) for key, val in extras.items()},
        )
        return StepResult(loss=loss, breakdown=breakdown)
    if isinstance(value, Mapping):
        payload = dict(value)
        loss = payload.pop("loss", None)
        if loss is None:
            raise ContractError(
                "a custom train_step returned a mapping without a 'loss' key"
            )
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(float(loss), dtype=torch.float32, requires_grad=True)
        breakdown = payload.pop("breakdown", None)
        if not isinstance(breakdown, LossBreakdown):
            breakdown = LossBreakdown(
                total=float(loss.detach().item()),
                extras={str(key): float(val) for key, val in payload.items()
                        if isinstance(val, (int, float))},
            )
        return StepResult(
            loss=loss,
            breakdown=breakdown,
            stepped=bool(payload.get("stepped", True)),
        )
    where = f" for node {getattr(node, 'path', node)!r}" if node is not None else ""
    raise ContractError(
        f"a custom train_step{where} returned {type(value).__name__}; expected a "
        "StepResult, a loss tensor, a (loss, dict) tuple, or a mapping with 'loss'"
    )
