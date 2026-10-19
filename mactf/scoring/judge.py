"""A deterministic task judge.

Training needs a signal. framework.md does not permit rewarding unverified
claims, so this judge only records rules that the task itself can verify:

* **R1** honest task completion, magnitude = accuracy on the task's checks.
* **R2** verified honesty, magnitude = calibration margin (the probability mass
  placed on the correct class relative to uniform).
* **R14** graceful degradation, when accuracy is partial but above chance.

It deliberately records **no** penalty for a wrong answer: framework.md penalizes
dishonesty, not error, so an incorrect-but-honestly-labelled prediction earns
nothing instead of being punished. Penalties come from the red-team injectors or
from an external evaluator supplying its own ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ledger import Ledger


@dataclass
class JudgeReport:
    peer: str
    accuracy: float
    calibration: float
    recorded: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"judge[{self.peer}] accuracy={self.accuracy:.3f} "
            f"calibration={self.calibration:.3f} rules={list(self.recorded)}"
        )


class TaskJudge:
    """Scores peer proposals against a verifiable task target."""

    def __init__(
        self,
        *,
        completion_threshold: float = 0.5,
        partial_threshold: float = 0.25,
        calibration_floor: float = 0.0,
    ) -> None:
        self.completion_threshold = float(completion_threshold)
        self.partial_threshold = float(partial_threshold)
        self.calibration_floor = float(calibration_floor)

    def judge(
        self,
        ledger: Ledger,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        peer: str = "",
    ) -> JudgeReport:
        if logits is None or targets is None:
            return JudgeReport(peer=peer, accuracy=0.0, calibration=0.0, recorded=())

        probabilities = torch.softmax(logits.float(), dim=-1)
        batch = min(probabilities.shape[0], targets.shape[0])
        if batch == 0:
            return JudgeReport(peer=peer, accuracy=0.0, calibration=0.0, recorded=())

        predictions = probabilities[:batch].argmax(dim=-1)
        labels = targets[:batch].to(predictions.device)
        accuracy = float((predictions == labels).float().mean().item())

        # Calibration: how much probability mass the correct class got above uniform.
        uniform = 1.0 / max(1, probabilities.shape[-1])
        correct_mass = probabilities[:batch].gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1).mean()
        calibration = float((correct_mass - uniform).clamp_min(0.0).item())

        recorded: list[str] = []
        if accuracy >= self.completion_threshold:
            ledger.record(
                "R1",
                magnitude=accuracy,
                evidence=f"task checks: accuracy={accuracy:.3f}",
                source="task_judge",
            )
            recorded.append("R1")
        elif accuracy >= self.partial_threshold:
            ledger.record(
                "R14",
                magnitude=accuracy,
                evidence=f"partial accuracy={accuracy:.3f}",
                source="task_judge",
            )
            recorded.append("R14")

        if calibration > self.calibration_floor and accuracy > 0.0:
            ledger.record(
                "R2",
                magnitude=calibration,
                evidence=f"calibration margin={calibration:.3f}",
                source="task_judge",
            )
            recorded.append("R2")

        return JudgeReport(
            peer=peer,
            accuracy=accuracy,
            calibration=calibration,
            recorded=tuple(recorded),
        )
