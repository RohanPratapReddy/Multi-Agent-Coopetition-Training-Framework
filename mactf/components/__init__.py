"""The unified component model (document 17 sections 3-5).

Every pluggable part of training is a *component* of a declared *kind*, registered under a
name with an origin and a mutability class, resolved from a layered spec. This package
imports nothing heavy: ``import mactf.components`` pulls in neither torch nor any
framework package; contracts import the classes they check lazily.
"""

from __future__ import annotations

from . import contracts as _contracts
from .decorators import (
    get_component,
    register,
    register_aggregator,
    register_function,
    register_optimizer,
    register_peer_factory,
    register_peer_strategy,
    register_policy,
    register_reward_term,
    register_scheduler,
    register_scorer,
    register_weight_update,
    unregister,
)
from .kinds import (
    KIND_SPECS,
    RL_FAMILIES,
    Boundary,
    ComponentKind,
    KindSpec,
    Mutability,
    Origin,
    all_kind_ids,
    parse_kind_id,
)
from .legacy import LegacyTable
from .params import MISSING, ParamSpec
from .plugins import (
    ENTRY_POINT_GROUP,
    LoadReport,
    ensure_components_loaded,
    load_components,
    policy_from_settings,
)
from .reflect import Reflection, canonical_json, constructor_keys, reflect, source_digest, spec_hash
from .registry import (
    HUB,
    Component,
    ComponentRegistry,
    Hub,
    HubPolicy,
    Overlay,
)
from .resolver import (
    LAYERS,
    ComponentResolver,
    ComponentSpec,
    LayerResult,
    Resolved,
    merge_layers,
)
from .slots import Slot, SlotTable, SwapRecord

_contracts.install()


def overlay(label: str = "run", *, policy: HubPolicy | None = None):
    """A per-run registry overlay: ``with overlay() as scope: ...`` (see ``Hub.scoped``)."""
    return HUB.scoped(label, policy=policy)


__all__ = [
    "Boundary",
    "Component",
    "ComponentKind",
    "ComponentRegistry",
    "ComponentResolver",
    "ComponentSpec",
    "ENTRY_POINT_GROUP",
    "HUB",
    "Hub",
    "HubPolicy",
    "KIND_SPECS",
    "KindSpec",
    "LAYERS",
    "LayerResult",
    "LegacyTable",
    "LoadReport",
    "MISSING",
    "Mutability",
    "Origin",
    "Overlay",
    "ParamSpec",
    "RL_FAMILIES",
    "Reflection",
    "Resolved",
    "Slot",
    "SlotTable",
    "SwapRecord",
    "all_kind_ids",
    "canonical_json",
    "constructor_keys",
    "ensure_components_loaded",
    "get_component",
    "load_components",
    "merge_layers",
    "overlay",
    "parse_kind_id",
    "policy_from_settings",
    "reflect",
    "register",
    "register_aggregator",
    "register_function",
    "register_optimizer",
    "register_peer_factory",
    "register_peer_strategy",
    "register_policy",
    "register_reward_term",
    "register_scheduler",
    "register_scorer",
    "register_weight_update",
    "source_digest",
    "spec_hash",
    "unregister",
]
