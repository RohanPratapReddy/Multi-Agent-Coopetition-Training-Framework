"""The checkpoint section for contrastive distillation (document 14 section 11).

``DistillationState`` is deliberately small. Teacher *weights* are never stored: a
teacher may be a multi-gigabyte model on a shared volume, and re-serialising it
with every checkpoint is not a design anyone wants. What is stored is everything
needed to prove a resumed run is the same run: teacher identities, the schedule
cursor, the sigma history, and the conformance record (trust, profiles, audit
digests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "DistillResumeReport",
    "STATE_FORMAT_VERSION",
    "STATE_HISTORY",
    "DistillationState",
]

STATE_HISTORY = 512
STATE_FORMAT_VERSION = 2


@dataclass
class DistillationState:
    """What one checkpoint records about the distillation policy."""

    enabled: bool = False
    teacher_names: tuple[str, ...] = ()
    teacher_digests: Mapping[str, str] = field(default_factory=dict)
    step: int = 0
    bake: int = 0
    aggregations: int = 0
    blends: int = 0
    sigma_state: Mapping[str, Any] = field(default_factory=dict)
    view_descriptions: Mapping[str, Any] = field(default_factory=dict)
    trust_state: Mapping[str, Any] = field(default_factory=dict)
    trust_profiles: Mapping[str, Any] = field(default_factory=dict)
    audit_digests: Mapping[str, str] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    loss_history: list[dict[str, float]] = field(default_factory=list)
    episode_counts: Mapping[str, int] = field(default_factory=dict)

    def record(self, row: Mapping[str, float]) -> None:
        """Append one compact numeric row, keeping at most ``STATE_HISTORY``."""
        self.loss_history.append({str(key): float(value) for key, value in row.items()})
        overflow = len(self.loss_history) - STATE_HISTORY
        if overflow > 0:
            del self.loss_history[:overflow]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": STATE_FORMAT_VERSION,
            "enabled": bool(self.enabled),
            "teacher_names": list(self.teacher_names),
            "teacher_digests": dict(self.teacher_digests),
            "step": int(self.step),
            "bake": int(self.bake),
            "aggregations": int(self.aggregations),
            "blends": int(self.blends),
            "sigma_state": dict(self.sigma_state),
            "view_descriptions": dict(self.view_descriptions),
            "trust_state": dict(self.trust_state),
            "trust_profiles": dict(self.trust_profiles),
            "audit_digests": dict(self.audit_digests),
            "config": dict(self.config),
            "episode_counts": {str(k): int(v) for k, v in self.episode_counts.items()},
            "loss_history": [dict(row) for row in self.loss_history][-STATE_HISTORY:],
        }


@dataclass(frozen=True)
class DistillResumeReport:
    """What restoring the distillation section did, and every discrepancy found.

    A digest mismatch is a *note*, never an exception: resuming against a
    retrained teacher is a legitimate operation, and refusing it would force the
    user to delete a checkpoint to continue. What must not happen is the mismatch
    going unnoticed.
    """

    section: str = "distillation"
    restored: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "restored": list(self.restored),
            "skipped": list(self.skipped),
            "notes": list(self.notes),
        }
