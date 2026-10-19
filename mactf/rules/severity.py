"""Severity tiers and proportionality (framework.md section 4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from ..exceptions import ComplianceError


class Severity(IntEnum):
    """Ordered severity tiers S0..S4 from framework.md section 4.1."""

    NONE = 0
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    SEVERE = 4

    @property
    def label(self) -> str:
        return f"S{int(self)}"

    @classmethod
    def parse(cls, value: Any) -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            if 0 <= value <= 4:
                return cls(value)
            raise ValueError(f"severity integer out of range: {value}")
        text = str(value).strip().upper()
        if text.startswith("S") and text[1:].isdigit():
            number = int(text[1:])
            if 0 <= number <= 4:
                return cls(number)
        names = {
            "NONE": cls.NONE,
            "MINOR": cls.MINOR,
            "MODERATE": cls.MODERATE,
            "MAJOR": cls.MAJOR,
            "SEVERE": cls.SEVERE,
        }
        if text in names:
            return names[text]
        raise ValueError(f"unrecognised severity: {value!r}")


@dataclass(frozen=True)
class Proportionality:
    """The four judgments of framework.md section 4.2.

    Severity is chosen from these factors rather than guessed. ``scope`` and
    ``reversibility`` are 0..1 risk fractions, ``intent`` is an enum-like
    string, and ``repetition`` is the occurrence index (1 = first).
    """

    scope: float = 0.5
    reversibility: float = 0.5
    intent: str = "negligent"
    repetition: int = 1

    INTENT_RISK = {
        "negligent": 0.5,
        "accidental": 0.4,
        "reckless": 0.8,
        "deliberate": 1.0,
    }

    def __post_init__(self) -> None:
        if not 0.0 <= self.scope <= 1.0:
            raise ValueError("scope must be within [0, 1]")
        if not 0.0 <= self.reversibility <= 1.0:
            raise ValueError("reversibility must be within [0, 1]")
        if self.intent not in self.INTENT_RISK:
            raise ValueError(
                f"intent must be one of {sorted(self.INTENT_RISK)}, got {self.intent!r}"
            )
        if self.repetition < 1:
            raise ValueError("repetition must be >= 1")

    @property
    def risk(self) -> float:
        """Normalized impact in [0, 1]; irreversibility is the dominant term."""
        intent = self.INTENT_RISK[self.intent]
        repeat = 1.0 - 0.5 ** (self.repetition - 1)
        irr = 1.0 - self.reversibility
        return min(1.0, max(0.0, 0.35 * self.scope + 0.35 * irr + 0.30 * intent) + 0.15 * repeat)

    def suggest(self, *, ceiling: Severity = Severity.SEVERE) -> Severity:
        """Map the risk fraction onto a tier.

        Thresholds are ordering guarantees, not point values: the mapping only
        has to be monotone in risk and bounded by ``ceiling``.
        """
        risk = self.risk
        if risk < 0.15:
            tier = Severity.NONE
        elif risk < 0.35:
            tier = Severity.MINOR
        elif risk < 0.55:
            tier = Severity.MODERATE
        elif risk < 0.80:
            tier = Severity.MAJOR
        else:
            tier = Severity.SEVERE
        return Severity(min(int(tier), int(ceiling)))


def escalate(severity: Severity, *, max_tier: int = 3) -> Severity:
    """Raise a severity by exactly one tier, never above ``max_tier``.

    framework.md section 8.3 caps warning escalation at S3 and forbids
    escalation into the severe, non-warning rules.
    """
    return Severity(min(int(severity) + 1, max_tier))


def weight_for(severity: Severity, weights: dict[str, Any]) -> float:
    """Look up the configured magnitude for a severity tier."""
    key = severity.label
    if key not in weights:
        raise ComplianceError(f"SEVERITY.WEIGHTS has no entry for {key}")
    return float(weights[key])


def check_monotone(weights: dict[str, Any]) -> None:
    """Assert the configured magnitudes increase with severity."""
    ordered = [float(weights[f"S{i}"]) for i in range(5)]
    if ordered != sorted(ordered):
        raise ComplianceError("SEVERITY.WEIGHTS must be non-decreasing across S0..S4")
