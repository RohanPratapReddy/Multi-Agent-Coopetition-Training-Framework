"""KappaPlan: per-group temperature settings (document 17 section 10.7)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..coopetition.temperature import AdaptiveKappa
from ..exceptions import ImproperlyConfigured
from .weight_plan import ROOT_LABEL, pick_glob

KAPPA_KEYS = (
    "KAPPA_INIT", "KAPPA_MIN", "KAPPA_MAX",
    "TARGET_ENTROPY_RATIO", "BETA_MOMENTUM", "ETA_KAPPA",
)
_FORBIDDEN = {
    "SIGNAL_SOURCE": "one score range per tree",
    "WEIGHT_FLOOR": "the floor is one number for the run",
    "WEIGHT_CAP": "the cap is one number for the run",
}


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ImproperlyConfigured(f"E062: {where} must be a finite number, got {value!r}")
    return float(value)


class KappaPlan:
    """Per-group ``AdaptiveKappa`` settings: the base block plus the most specific override."""

    def __init__(
        self, overrides: Mapping[str, Mapping[str, Any]] | None = None, *, strict_globs: bool = True
    ) -> None:
        self.strict_globs = bool(strict_globs)
        self._overrides: dict[str, dict[str, float]] = {}
        for glob, block in dict(overrides or {}).items():
            self._overrides[str(glob)] = self._check(str(glob), block)

    @classmethod
    def from_settings(cls, coopetition: Mapping[str, Any]) -> "KappaPlan":
        raw = coopetition.get("KAPPA_OVERRIDES") or {}
        if not isinstance(raw, Mapping):
            raise ImproperlyConfigured("E062: COOPETITION.KAPPA_OVERRIDES must be a mapping of glob to block")
        return cls(raw, strict_globs=bool(coopetition.get("KAPPA_STRICT_GLOBS", True)))

    @staticmethod
    def _check(glob: str, block: Any) -> dict[str, float]:
        where = f"COOPETITION.KAPPA_OVERRIDES[{glob!r}]"
        if not glob or glob != glob.strip():
            raise ImproperlyConfigured(f"E062: invalid glob {glob!r} in COOPETITION.KAPPA_OVERRIDES")
        if not isinstance(block, Mapping) or not block:
            raise ImproperlyConfigured(f"E062: {where} must be a non-empty mapping")
        clean: dict[str, float] = {}
        for key, value in block.items():
            name = str(key).upper()
            if name in _FORBIDDEN:
                raise ImproperlyConfigured(f"E062: {where} may not set {name} ({_FORBIDDEN[name]})")
            if name not in KAPPA_KEYS:
                raise ImproperlyConfigured(f"E062: {where} has unknown key {name}; accepted: {list(KAPPA_KEYS)}")
            clean[name] = _number(value, f"{where}.{name}")
        return clean

    @property
    def overrides(self) -> dict[str, dict[str, float]]:
        return {glob: dict(block) for glob, block in self._overrides.items()}

    def settings_for(self, label: str, base: Mapping[str, Any]) -> dict[str, Any]:
        """``base`` with the winning override applied (``E062`` on a tie or a bad merge)."""
        merged = dict(base)
        glob = pick_glob(label, list(self._overrides), what="KAPPA_OVERRIDES", code="E062")
        if glob is not None:
            merged.update(self._overrides[glob])
        low, high = float(merged.get("KAPPA_MIN", 0.1)), float(merged.get("KAPPA_MAX", 10.0))
        if low <= 0 or high < low:
            raise ImproperlyConfigured(
                f"E062: group {label!r}: need 0 < KAPPA_MIN <= KAPPA_MAX, got {low:g} and {high:g}"
            )
        ratio = float(merged.get("TARGET_ENTROPY_RATIO", 0.8))
        if not 0.0 < ratio <= 1.0:
            raise ImproperlyConfigured(f"E062: group {label!r}: TARGET_ENTROPY_RATIO must be in (0, 1], got {ratio:g}")
        beta = float(merged.get("BETA_MOMENTUM", 0.9))
        if not 0.0 <= beta < 1.0:
            raise ImproperlyConfigured(f"E062: group {label!r}: BETA_MOMENTUM must be in [0, 1), got {beta:g}")
        return merged

    def build(
        self,
        label: str,
        base: Mapping[str, Any],
        *,
        num_peers: int,
        mean_abs_reference_signal: float | None = None,
    ) -> AdaptiveKappa:
        """The controller of group ``label``: ``AdaptiveKappa.from_settings`` on its merged block."""
        return AdaptiveKappa.from_settings(
            self.settings_for(label, base),
            num_peers=max(1, int(num_peers)),
            mean_abs_reference_signal=mean_abs_reference_signal,
        )

    def validate(self, labels: Sequence[str], *, initial: bool) -> list[str]:
        """``E062`` for a tie or an unmatched glob at build; ``W022`` for an unmatched glob later."""
        import fnmatch

        notes: list[str] = []
        for label in labels:
            self.settings_for(label, {})                      # E062 on ties and bad merges
        for glob in self._overrides:
            if any(label == glob or fnmatch.fnmatchcase(label, glob) for label in labels):
                continue
            message = f"KAPPA_OVERRIDES {glob!r} matches no aggregation group; groups are {list(labels)}"
            if initial and self.strict_globs:
                raise ImproperlyConfigured(f"E062: {message}")
            notes.append(f"W022: {message}")
        return notes
