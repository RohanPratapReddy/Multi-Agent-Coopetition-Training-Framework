"""Training layer: optimizers, trainer, strategies, logging, and watchdogs.

Document 08 and 09. This package may import :mod:`mactf.rl`, :mod:`mactf.scoring`,
:mod:`mactf.coopetition`, :mod:`mactf.runtime`, and
:mod:`mactf.orchestration.checkpoint`. It must not import :mod:`mactf.http`,
:mod:`mactf.contrib.api`, or :mod:`mactf.client`.

Public names resolve lazily (PEP 562) so ``import mactf.training.logging`` does not
import ``torch``; ``from mactf.training import MACTFTrainer`` still works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import lazy_exports

_EXPORTS: dict[str, str] = {
    "EpochResult": "protocol", "StepResult": "protocol", "TrainResult": "protocol",
    "TrainerCapabilities": "protocol", "TrainerProtocol": "protocol",
    "MACTFTrainer": "trainer", "TrainSpec": "trainer",
}

if TYPE_CHECKING:  # pragma: no cover
    from .protocol import EpochResult, StepResult, TrainResult, TrainerCapabilities, TrainerProtocol
    from .trainer import MACTFTrainer, TrainSpec

__all__: list[str] = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
