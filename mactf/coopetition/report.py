"""The aggregation audit record, kept free of torch so a client can parse it.

``AggregationReport`` is what ``Aggregator.aggregate`` explains itself with and what
``GET /runs/{id}/aggregations`` returns. It lives apart from ``aggregation.py``
(which imports torch for the maths) so ``mactf.client`` can rebuild one from JSON
without loading a 2 GB package; ``aggregation`` re-exports it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..wire import floats, need, opt, strings


@dataclass
class AggregationReport:
    """Full derivation of one aggregation step."""

    weights: dict[str, float]
    raw_weights: dict[str, float]
    kappa: float
    entropy: float
    adjustments: list[str] = field(default_factory=list)
    fallback: bool = False
    gated_peers: tuple[str, ...] = ()
    trimmed_peers: tuple[str, ...] = ()
    order: tuple[str, ...] = ()
    #: Which score range produced these weights (document 11 section 8.5).
    #: ``final_score`` is the legacy training signal; ``points`` is the bounded
    #: calibrated signal. Recording it is what makes an old run explainable.
    signal_source: str = "final_score"
    #: The per-peer calibration that produced ``signal_source="points"``.
    normalization: Any | None = None

    def describe(self) -> str:
        pairs = ", ".join(f"{peer}={weight:.4f}" for peer, weight in self.weights.items())
        detail = f"kappa={self.kappa:.4f} H={self.entropy:.4f} weights[{pairs}]"
        if self.adjustments:
            detail += " adjustments[" + "; ".join(self.adjustments) + "]"
        if self.fallback:
            detail += " FALLBACK=reference"
        return detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "raw_weights": dict(self.raw_weights),
            "kappa": self.kappa,
            "entropy": self.entropy,
            "adjustments": list(self.adjustments),
            "fallback": self.fallback,
            "gated_peers": list(self.gated_peers),
            "trimmed_peers": list(self.trimmed_peers),
            "order": list(self.order),
            "signal_source": self.signal_source,
            "normalization": (
                self.normalization.as_dict()
                if hasattr(self.normalization, "as_dict")
                else self.normalization
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AggregationReport":
        """Inverse of :meth:`as_dict`: lists become the declared tuples again."""
        from ..scoring.normalize import NormalizationReport

        owner = "AggregationReport"
        normalization = payload.get("normalization")
        return cls(
            weights=floats(need(owner, payload, "weights")),
            raw_weights=floats(need(owner, payload, "raw_weights")),
            kappa=float(need(owner, payload, "kappa")),
            entropy=float(need(owner, payload, "entropy")),
            adjustments=[str(item) for item in (payload.get("adjustments") or [])],
            fallback=bool(opt(payload, "fallback", False)),
            gated_peers=strings(payload.get("gated_peers")),
            trimmed_peers=strings(payload.get("trimmed_peers")),
            order=strings(payload.get("order")),
            signal_source=str(opt(payload, "signal_source", "final_score")),
            normalization=(
                NormalizationReport.from_dict(normalization) if isinstance(normalization, dict) else normalization
            ),
        )
