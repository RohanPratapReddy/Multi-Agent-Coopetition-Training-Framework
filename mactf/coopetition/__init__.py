"""Coopetition: peer adapters, adaptive temperature, and constrained aggregation."""

from __future__ import annotations

from .adapter import PeerAdapter, PeerContract, infer_contract
from .aggregation import AggregateResult, AggregationReport, Aggregator, feasible_bounds
from .peer import Peer, build_peer, build_reference, clone_peer, require_same_layout
from .temperature import AdaptiveKappa, TemperatureState, entropy_collapsed

__all__ = [
    "AdaptiveKappa",
    "AggregateResult",
    "AggregationReport",
    "Aggregator",
    "Peer",
    "PeerAdapter",
    "PeerContract",
    "TemperatureState",
    "build_peer",
    "build_reference",
    "clone_peer",
    "entropy_collapsed",
    "feasible_bounds",
    "infer_contract",
    "require_same_layout",
]
