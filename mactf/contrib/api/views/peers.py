"""Peers and weight preview (document 02 section 3.6)."""

from __future__ import annotations

from ....coopetition.adapter import PeerAdapter
from ....coopetition.introspect import (
    build_swarm,
    describe_contract,
    describe_peer,
    preview_weights,
)
from ....exceptions import InvalidArgument
from ....http.exceptions import Http404, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from .. import deps
from ..serializers import WeightPreviewRequestSerializer
from .base import ApiView


def swarm(request: Request) -> tuple[dict[str, PeerAdapter], str]:
    """The peers of the one live run when there is exactly one, else the configured ones."""
    manager = deps.get_run_manager_dep(request)
    live = manager.live_run_ids()
    if len(live) == 1:
        handle = manager.get_handle(live[0])
        if handle is not None and handle.orchestrator is not None and handle.orchestrator.peers:
            return {name: peer.adapter for name, peer in handle.orchestrator.peers.items()}, f"run:{live[0]}"
    return build_swarm(deps.get_settings_dep(request)), "settings"


def _find(request: Request, name: str) -> tuple[PeerAdapter, str]:
    adapters, source = swarm(request)
    if name not in adapters:
        raise Http404(f"no peer {name!r}; known peers: {sorted(adapters)}")
    return adapters[name], source


class PeerListView(ApiView):
    summary = "The configured or live peers"

    def get(self, request: Request) -> JsonResponse:
        adapters, source = swarm(request)
        return JsonResponse([describe_peer(name, adapter, source) for name, adapter in adapters.items()])


class PeerDetailView(ApiView):
    summary = "One peer with its contract"

    def get(self, request: Request, name: str) -> JsonResponse:
        adapter, source = _find(request, name)
        return JsonResponse({**describe_peer(name, adapter, source), **describe_contract(adapter)})


class PeerContractView(ApiView):
    summary = "A peer's input/output contract"

    def get(self, request: Request, name: str) -> JsonResponse:
        adapter, _ = _find(request, name)
        return JsonResponse(describe_contract(adapter))


class WeightPreviewView(ApiView):
    scopes = {"post": "read"}
    summary = "What would the swarm's weights be for these scores?"
    request_serializer = {"post": WeightPreviewRequestSerializer}
    description = (
        "Builds an aggregator from COOPETITION settings with num_peers=len(scores). A throwaway "
        "controller is used, so a preview never moves a run's kappa."
    )

    def post(self, request: Request) -> JsonResponse:
        body = self.body(request, WeightPreviewRequestSerializer)
        try:
            report = preview_weights(
                deps.get_settings_dep(request), body["scores"],
                eligible=body.get("eligible"), confidence=body.get("confidence"),
                advance=bool(body.get("advance", False)),
            )
        except InvalidArgument as exc:
            raise ValidationError({exc.field or "scores": [str(exc)]}) from None
        return JsonResponse(report)
