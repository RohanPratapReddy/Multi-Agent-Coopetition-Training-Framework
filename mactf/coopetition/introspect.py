"""Peer and weight introspection with no HTTP in it (document 16 section 7.6).

The API's peer views and the CLI's ``peers`` command both call these functions, so a peer
looks the same on both surfaces. Importing this module is cheap: ``torch`` (via the peer
adapter and the orchestrator's module factory) is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Mapping

from ..exceptions import InvalidArgument

if TYPE_CHECKING:  # pragma: no cover
    from .adapter import PeerAdapter

__all__ = ["build_swarm", "describe_contract", "describe_peer", "preview_weights"]


def build_swarm(settings: Any) -> dict[str, "PeerAdapter"]:
    """Adapters for the configured peers (``_default_modules``), without building a run."""
    from ..orchestration.orchestrator import _default_modules
    from .adapter import PeerAdapter

    training = settings["TRAINING"]
    return {
        name: PeerAdapter(
            module, name=name,
            action_dim=int(training["ACTION_DIM"]), state_dim=int(training["STATE_DIM"]),
        )
        for name, module in _default_modules(settings).items()
    }


def describe_peer(name: str, adapter: "PeerAdapter", source: str) -> dict[str, Any]:
    """``{"name", "class", "parameter_count", "layout_hash", "source"}``.

    ``layout_hash`` is the first 16 hex digits of ``sha256(repr(parameter_shapes()))``.
    """
    layout = hashlib.sha256(repr(adapter.parameter_shapes()).encode("utf-8")).hexdigest()
    return {
        "name": name,
        "class": type(adapter.module).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in adapter.parameters())),
        "layout_hash": layout[:16],
        "source": source,
    }


def describe_contract(adapter: "PeerAdapter") -> dict[str, Any]:
    """``{"contract", "input_names", "outputs_value", "output_kind", "notes"}``."""
    contract = adapter.contract
    return {
        "contract": contract.describe(),
        "input_names": list(contract.input_names),
        "outputs_value": bool(contract.outputs_value),
        "output_kind": contract.output_kind,
        "notes": list(contract.notes),
    }


def preview_weights(
    settings: Any,
    scores: Mapping[str, float],
    *,
    eligible: Mapping[str, bool] | None = None,
    confidence: Mapping[str, float] | None = None,
    advance: bool = False,
) -> dict[str, Any]:
    """What the swarm's weights would be for ``scores``: ``WeightReport.as_dict()``.

    Builds an ``Aggregator`` from ``COOPETITION`` with ``num_peers=len(scores)``; a throwaway
    controller is used, so a preview never moves a run's kappa. ``InvalidArgument`` for an
    empty ``scores`` or a missing ``eligible``/``confidence`` entry (``field`` names the
    mapping); ``ComplianceError`` when the floor/cap cannot be satisfied for that many peers.
    """
    from .aggregation import Aggregator, feasible_bounds

    names = list(scores)
    if not names:
        raise InvalidArgument("at least one score is required.", field="scores")

    def aligned(mapping: Mapping[str, Any] | None, key: str, label: str) -> list[Any] | None:
        if mapping is None:
            return None
        missing = [name for name in names if name not in mapping]
        if missing:
            raise InvalidArgument(f"missing {label} for peer(s) {missing}.", field=key)
        return [mapping[name] for name in names]

    eligible_list = aligned(eligible, "eligible", "eligibility")
    confidence_list = aligned(confidence, "confidence", "confidence")
    coop = settings["COOPETITION"]
    feasible_bounds(len(names), floor=float(coop["WEIGHT_FLOOR"]), cap=float(coop["WEIGHT_CAP"]))
    aggregator = Aggregator.from_settings(coop, num_peers=len(names))
    report = aggregator.compute_weights(
        names,
        [float(scores[name]) for name in names],
        eligible=eligible_list,
        confidence=confidence_list,
        advance=bool(advance),
    )
    return report.as_dict()
