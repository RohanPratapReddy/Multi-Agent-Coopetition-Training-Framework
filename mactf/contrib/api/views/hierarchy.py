"""Peer trees and hierarchical inference (document 02 section 3.9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ....coopetition.tree import PeerTree, validate_path
from ....http.exceptions import Http404, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from ....orchestration.live import jsonable
from ....runtime import load_checkpoint
from ....scoring.hierarchy import run_inference
from .. import deps
from ..serializers import (
    InferenceRequestSerializer,
    InferenceResultSerializer,
    PeerNodeSerializer,
    PeerTreeSerializer,
)
from .base import ApiView, conflict, get_run_or_404


def checkpoint_tree(request: Request, run_id: str) -> PeerTree | None:
    """The tree in the newest readable checkpoint (``tree`` section, else its ``peers`` keys)."""
    for row in reversed(deps.get_registry_dep(request).checkpoints(run_id)):
        path = Path(row["path"])
        if not path.exists():
            continue
        try:
            payload = load_checkpoint(path)
        except Exception:  # noqa: BLE001 - try the next-newest file
            continue
        section = payload.get("tree") or payload.get("topology")
        try:
            return PeerTree.from_dict(section if isinstance(section, dict) and section.get("nodes") else payload)
        except Exception:  # noqa: BLE001
            continue
    return None


def resolve_tree(request: Request, run_id: str) -> tuple[PeerTree | None, str]:
    """``(tree, source)``: the live tree when the manager holds the run, else the checkpoint's."""
    live = deps.get_run_manager_dep(request).tree(run_id)
    if live is not None:
        return live, "live"
    stored = checkpoint_tree(request, run_id)
    if stored is not None:
        return stored, "checkpoint"
    return None, ""


def tree_payload(tree: PeerTree, source: str) -> dict[str, Any]:
    payload = tree.to_dict()
    payload["source"] = source
    payload["depth"] = int(tree.depth)
    payload["node_count"] = len(payload["nodes"])
    return payload


def nearest_prefix(tree: PeerTree, path: str) -> str | None:
    parts = path.split("/")
    while parts:
        candidate = "/".join(parts)
        if candidate in tree:
            return candidate
        parts.pop()
    return None


def _node_or_404(tree: PeerTree, path: str) -> Any:
    try:
        validate_path(path)
    except Exception as exc:  # noqa: BLE001 - grammar errors are the client's: 422 via ContractError
        raise ValidationError({"path": [str(exc)]}) from None
    if path not in tree:
        raise Http404(f"no node {path!r} in the tree", extensions={"nearest_prefix": nearest_prefix(tree, path)})
    return tree[path]


def _tensorize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tensorize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            return [_tensorize(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(float(value))
    return value


def infer(request: Request, tree: PeerTree, body: dict[str, Any], node_path: str | None) -> dict[str, Any]:
    api = deps.get_settings_dep(request).get("API") or {}
    ceiling = int(api.get("INFERENCE_MAX_WORKERS", 4))
    if body["max_workers"] > ceiling:
        raise ValidationError({"max_workers": [f"must be at most API.INFERENCE_MAX_WORKERS={ceiling}."]})
    if node_path is not None:
        _node_or_404(tree, node_path)
    budget = body.get("budget") if body.get("budget") is not None else api.get("INFERENCE_BUDGET_DEFAULT")
    scorer = deps.get_scorer_dep(request) if body["score"] else None
    result = run_inference(
        tree,
        _tensorize(body["inputs"]),
        max_depth=body.get("max_depth"),
        budget=budget,
        parallel=body["parallel"],
        max_workers=body["max_workers"],
        score=body["score"],
        scorer=scorer,
        node_path=node_path,
        seed=body.get("seed"),
        persist_warnings=body["persist_warnings"],
    )
    return jsonable(result.as_dict())


class RunPeerTreeView(ApiView):
    summary = "The peer tree of a run"
    response_serializer = PeerTreeSerializer
    description = "The live tree when the run is in memory, else the newest checkpoint's tree; `source` says which."

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        tree, source = resolve_tree(request, run_id)
        if tree is None:
            raise Http404(
                f"run {run_id!r} has no loaded tree and no checkpoint", extensions={"reason": "no tree"}
            )
        return JsonResponse(tree_payload(tree, source))


class RunPeerNodeView(ApiView):
    summary = "One node of a run's tree with its latest scorecard"
    response_serializer = PeerNodeSerializer

    def get(self, request: Request, run_id: str, path: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        tree, source = resolve_tree(request, run_id)
        if tree is None:
            raise Http404(f"run {run_id!r} has no loaded tree and no checkpoint", extensions={"reason": "no tree"})
        node = _node_or_404(tree, path)
        registry = deps.get_registry_dep(request)
        latest = registry.latest_episode(run_id)
        card = None
        if latest is not None:
            row = registry.get_episode(run_id, latest)
            card = ((row or {}).get("scorecard") or {}).get("peers", {}).get(path)
        return JsonResponse({**node.describe(), "source": source, "latest_episode": latest, "scorecard": card})


class RunInferView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Run hierarchical inference over a run's tree"
    request_serializer = {"post": InferenceRequestSerializer}
    response_serializer = {"post": InferenceResultSerializer}
    description = "Budget exhaustion is a 422; a truncated result is never returned."

    def post(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        body = self.body(request, InferenceRequestSerializer)
        tree, _ = resolve_tree(request, run_id)
        if tree is None or not any(node.peer is not None for node in tree.walk()):
            raise conflict("run has no loaded tree and no checkpoint", reason="run has no loaded tree and no checkpoint")
        return JsonResponse(infer(request, tree, body, body.get("node_path")))


class PeerInferView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Run inference rooted at one node"
    request_serializer = {"post": InferenceRequestSerializer}
    response_serializer = {"post": InferenceResultSerializer}

    def post(self, request: Request, path: str) -> JsonResponse:
        body = self.body(request, InferenceRequestSerializer)
        manager = deps.get_run_manager_dep(request)
        run_id = request.GET.get("run_id")
        if run_id is None:
            live = manager.live_run_ids()
            if len(live) != 1:
                raise conflict(
                    "no run was named and there is not exactly one live run",
                    reason="pass ?run_id=", live_runs=live,
                )
            run_id = live[0]
        get_run_or_404(request, run_id)
        tree = manager.tree(run_id)
        if tree is None:
            raise conflict("the run has no live tree", reason="run has no live tree", run_id=run_id)
        return JsonResponse(infer(request, tree, body, path))
