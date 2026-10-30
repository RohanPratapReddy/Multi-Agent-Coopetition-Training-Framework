"""Coopetition: peers, the peer tree, delegation, synthesis, and aggregation.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Names resolve on first access (PEP 562): importing a light module such as
# ``mactf.scoring.scorecard`` must not drag torch in through this package.
_EXPORTS: dict[str, str] = {
    "AdaptiveKappa": "temperature",
    "AggregateResult": "aggregation",
    "AggregationReport": "aggregation",
    "Aggregator": "aggregation",
    "BestByScore": "synthesis",
    "BroadcastDelegation": "delegation",
    "COOPERATIVE_RULES": "pool",
    "ConcatSynthesis": "synthesis",
    "ConfidenceDelegation": "delegation",
    "ConsensusSynthesis": "synthesis",
    "CooperativeAllocation": "pool",
    "CooperativeBudget": "pool",
    "CooperativePool": "pool",
    "DELEGATION_TYPES": "delegation",
    "DelegationPolicy": "delegation",
    "DuplicateNodeError": "tree",
    "ExplicitDelegation": "delegation",
    "ExplicitSynthesis": "synthesis",
    "HierarchySpec": "tree",
    "LearnedSynthesizer": "synthesis",
    "LevelSpec": "tree",
    "NodeKind": "tree",
    "OrphanNodeError": "tree",
    "Peer": "peer",
    "PeerAdapter": "adapter",
    "PeerContract": "adapter",
    "PeerNode": "tree",
    "PeerTree": "tree",
    "RouteByRoleDelegation": "delegation",
    "SYNTHESIS_TYPES": "synthesis",
    "SelectKDelegation": "delegation",
    "SynthesisFunction": "synthesis",
    "TemperatureState": "temperature",
    "TreeCycleError": "tree",
    "TreeInconsistencyError": "tree",
    "TreeTooLargeError": "tree",
    "TreeValidationError": "tree",
    "WeightedConsensus": "synthesis",
    "aggregation_groups": "tree",
    "branch_void": "tree",
    "build_delegation": "delegation",
    "build_peer": "peer",
    "build_reference": "peer",
    "build_synthesis": "synthesis",
    "clone_peer": "peer",
    "cooperative_total": "pool",
    "delegation_capabilities": "delegation",
    "entropy_collapsed": "temperature",
    "feasible_bounds": "aggregation",
    "infer_contract": "adapter",
    "node_seed": "tree",
    "require_same_layout": "peer",
    "synthesis_capabilities": "synthesis",
    "validate_path": "tree",
}

if TYPE_CHECKING:  # pragma: no cover - static analysers see the real imports
    from .adapter import PeerAdapter, PeerContract, infer_contract
    from .aggregation import AggregateResult, AggregationReport, Aggregator, feasible_bounds
    from .delegation import BroadcastDelegation, ConfidenceDelegation, DELEGATION_TYPES, DelegationPolicy, ExplicitDelegation, RouteByRoleDelegation, SelectKDelegation, build_delegation, delegation_capabilities
    from .peer import Peer, build_peer, build_reference, clone_peer, require_same_layout
    from .pool import COOPERATIVE_RULES, CooperativeAllocation, CooperativeBudget, CooperativePool, cooperative_total
    from .synthesis import BestByScore, ConcatSynthesis, ConsensusSynthesis, ExplicitSynthesis, LearnedSynthesizer, SYNTHESIS_TYPES, SynthesisFunction, WeightedConsensus, build_synthesis, synthesis_capabilities
    from .temperature import AdaptiveKappa, TemperatureState, entropy_collapsed
    from .tree import DuplicateNodeError, HierarchySpec, LevelSpec, NodeKind, OrphanNodeError, PeerNode, PeerTree, TreeCycleError, TreeInconsistencyError, TreeTooLargeError, TreeValidationError, aggregation_groups, branch_void, node_seed, validate_path

__all__ = ["AdaptiveKappa", "AggregateResult", "AggregationReport", "Aggregator", "BestByScore", "BroadcastDelegation", "COOPERATIVE_RULES", "ConcatSynthesis", "ConfidenceDelegation", "ConsensusSynthesis", "CooperativeAllocation", "CooperativeBudget", "CooperativePool", "DELEGATION_TYPES", "DelegationPolicy", "DuplicateNodeError", "ExplicitDelegation", "ExplicitSynthesis", "HierarchySpec", "LearnedSynthesizer", "LevelSpec", "NodeKind", "OrphanNodeError", "Peer", "PeerAdapter", "PeerContract", "PeerNode", "PeerTree", "RouteByRoleDelegation", "SYNTHESIS_TYPES", "SelectKDelegation", "SynthesisFunction", "TemperatureState", "TreeCycleError", "TreeInconsistencyError", "TreeTooLargeError", "TreeValidationError", "WeightedConsensus", "aggregation_groups", "branch_void", "build_delegation", "build_peer", "build_reference", "build_synthesis", "clone_peer", "cooperative_total", "delegation_capabilities", "entropy_collapsed", "feasible_bounds", "infer_contract", "node_seed", "require_same_layout", "synthesis_capabilities", "validate_path"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
