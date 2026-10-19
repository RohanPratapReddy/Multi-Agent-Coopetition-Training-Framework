"""Checkpointing and rollback.

A checkpoint captures weights, optimizer state, kappa, warning registries, and the
episode counters, so a run is resumable. Rollback exists because framework.md
section 9/10 require a stable recovery path when aggregation degenerates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .. import signals
from ..exceptions import ComplianceError
from ..runtime import atomic_torch_save, load_checkpoint


@dataclass
class CheckpointRecord:
    path: Path
    episode: int
    run_id: str
    kappa: float
    entropy: float
    score: float
    created_at: float = field(default_factory=time.time)

    def describe(self) -> str:
        return (
            f"checkpoint ep={self.episode} kappa={self.kappa:.3f} "
            f"H={self.entropy:.3f} score={self.score:.4f} -> {self.path.name}"
        )


class CheckpointManager:
    """Writes, lists, and restores checkpoints for a run."""

    def __init__(self, directory: Path, *, run_id: str, keep: int = 5) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.keep = max(1, int(keep))

    def path_for(self, episode: int) -> Path:
        return self.directory / f"{self.run_id}-ep{episode:04d}.pt"

    def save(
        self,
        *,
        episode: int,
        peers: dict[str, Any],
        reference: Any | None,
        controller: Any | None,
        warnings: dict[str, Any] | None,
        runtime: dict[str, Any],
        kappa: float,
        entropy: float,
        score: float,
    ) -> CheckpointRecord:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "episode": episode,
            "created_at": time.time(),
            "runtime": runtime,
            "kappa": kappa,
            "entropy": entropy,
            "score": score,
            "peers": {name: peer.state_dict() for name, peer in peers.items()},
            "reference": reference.state_dict() if reference is not None else None,
            "controller": controller.state_dict() if controller is not None else None,
            "warnings": warnings or {},
        }
        path = atomic_torch_save(payload, self.path_for(episode))
        record = CheckpointRecord(
            path=path,
            episode=episode,
            run_id=self.run_id,
            kappa=kappa,
            entropy=entropy,
            score=score,
        )
        signals.checkpoint_written.send(sender=self, record=record, payload=payload)
        self._prune()
        return record

    def _prune(self) -> None:
        existing = sorted(self.directory.glob(f"{self.run_id}-ep*.pt"))
        for path in existing[: max(0, len(existing) - self.keep)]:
            path.unlink(missing_ok=True)

    def list(self) -> list[CheckpointRecord]:
        records: list[CheckpointRecord] = []
        for path in sorted(self.directory.glob(f"{self.run_id}-ep*.pt")):
            try:
                payload = load_checkpoint(path)
            except Exception:  # noqa: BLE001 - a corrupt file should not break listing
                continue
            records.append(
                CheckpointRecord(
                    path=path,
                    episode=int(payload.get("episode", 0)),
                    run_id=str(payload.get("run_id", self.run_id)),
                    kappa=float(payload.get("kappa", 0.0)),
                    entropy=float(payload.get("entropy", 0.0)),
                    score=float(payload.get("score", 0.0)),
                    created_at=float(payload.get("created_at", 0.0)),
                )
            )
        return records

    def latest(self) -> CheckpointRecord | None:
        records = self.list()
        return records[-1] if records else None

    def load(self, episode: int | None = None) -> dict[str, Any]:
        if episode is None:
            record = self.latest()
            if record is None:
                raise ComplianceError(f"no checkpoints found in {self.directory}")
            path = record.path
        else:
            path = self.path_for(episode)
        return load_checkpoint(path)

    def restore(
        self,
        payload: dict[str, Any],
        *,
        peers: dict[str, Any],
        reference: Any | None = None,
        controller: Any | None = None,
    ) -> int:
        """Restore peers and controller state, returning the restored episode."""
        for name, state in payload.get("peers", {}).items():
            if name in peers:
                peers[name].load_state_dict(state)
        if reference is not None and payload.get("reference"):
            reference.load_state_dict(payload["reference"])
        if controller is not None and payload.get("controller"):
            controller.load_state_dict(payload["controller"])
        episode = int(payload.get("episode", 0))
        signals.checkpoint_written.send(sender=self, record=None, payload=payload, restored=True)
        return episode


def should_rollback(
    *,
    entropy: float,
    max_entropy: float,
    collapse_ratio: float,
    fallback: bool,
    enabled: bool,
) -> tuple[bool, str]:
    """Decide whether the last aggregation must be rolled back."""
    if not enabled:
        return False, ""
    if fallback:
        return True, "aggregation fell back to the reference policy"
    if max_entropy > 0 and (entropy / max_entropy) < collapse_ratio:
        return True, (
            f"group entropy {entropy:.4f} fell below {collapse_ratio:.2%} of the "
            f"maximum {max_entropy:.4f}"
        )
    return False, ""
