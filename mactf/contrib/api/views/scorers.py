"""Scorers, trainer capabilities and trainer preview (document 02 section 3.8)."""

from __future__ import annotations

import copy
import importlib.util
from typing import Any

from ....exceptions import ContractError, ImproperlyConfigured, MACTFError
from ....http.exceptions import ValidationError
from ....http.request import Request
from ....http.response import JsonResponse, StreamingResponse, sse_event
from ....rl.functions.registry import function_capabilities
from ....scoring.registry import scorer_capabilities
from .. import deps
from ..serializers import (
    ScorerCapabilitySerializer,
    TrainerCapabilitiesSerializer,
    TrainerSpecSerializer,
)
from .base import ApiView, bool_param

EXTRAS = ("scipy", "peft", "transformers", "bitsandbytes", "tensorboard", "wandb")


def available_extras() -> dict[str, bool]:
    """Import probes: ``find_spec`` only, so probing never imports a heavy package."""
    found: dict[str, bool] = {}
    for name in EXTRAS:
        try:
            found[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found[name] = False
    return found


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    return value


def trainer_kwargs(spec: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Translate a validated ``TrainerSpec`` into ``MACTFTrainer`` keyword arguments.

    The model source (``model_factory`` or ``hierarchy``) is the caller's business:
    a spec describes *how* to train, never *what* is being trained.
    """
    from ....coopetition.tree import HierarchySpec
    from ....conf import Settings

    kwargs: dict[str, Any] = {}
    for key in (
        "num_peers", "peer_names", "optimizer", "loss", "finetuning", "peer_strategy",
        "weight_update", "max_grad_norm", "steps_per_epoch", "distillation",
    ):
        if spec.get(key) is not None:
            kwargs[key] = spec[key]
    if isinstance(kwargs.get("distillation"), dict):
        # Unset serializer fields arrive as None; for this block an unset key means
        # "defer to settings", so None must not reach the merge as a stated value.
        kwargs["distillation"] = _drop_none(kwargs["distillation"])
    if spec.get("grad_accumulation"):
        kwargs["grad_accumulation"] = int(spec["grad_accumulation"])
    if spec.get("hierarchy") is not None:
        kwargs["hierarchy"] = HierarchySpec.from_spec(spec["hierarchy"])
        kwargs.pop("num_peers", None)
    chain = spec.get("scorers")
    effective = settings
    if chain:
        stages = []
        for stage in chain.get("scorers") or []:
            entry: dict[str, Any] = {"type": stage["type"], **dict(stage.get("config") or {})}
            if stage.get("order") is not None:
                entry["order"] = stage["order"]
            stages.append(entry)
        if stages:
            kwargs["scorers"] = stages
        if chain.get("severity_weights"):
            merged = copy.deepcopy(settings.as_dict())
            merged.setdefault("SEVERITY", {})["WEIGHTS"] = dict(chain["severity_weights"])
            effective = Settings(merged)
    kwargs["settings"] = effective
    return kwargs


def _loss_types() -> list[str]:
    return [entry["type"] for entry in function_capabilities().get("loss", [])]


def capabilities_payload(request: Request) -> dict[str, Any]:
    from ....training import optimizer as opt

    backends = []
    granularity: dict[str, str] = {}
    for cls in (opt.TorchOptimizer, opt.ScipyOptimizer, opt.CallableOptimizer, opt.ParamGroupRouter):
        name = str(getattr(cls, "name", cls.__name__))
        step = "epoch" if cls is opt.ScipyOptimizer else "step"
        granularity[name] = step
        backends.append({"name": name, "class": cls.__name__, "granularity": step})
    scorers = [
        {
            "type": item["type"],
            "order": int(item["order"]),
            "required_context": list(item["requires"]),
            "config_keys": [],
            "description": f"{item['class']} ({item['kind']}{', root only' if item['root_only'] else ''})",
        }
        for item in scorer_capabilities()
    ]
    api = deps.get_settings_dep(request).get("API") or {}
    return {
        "optimizer_backends": backends,
        "optimizer_granularity": granularity,
        "loss_types": _loss_types(),
        "scorer_types": scorers,
        "finetune_types": ["lora", "adalora", "prefix", "qlora"],
        "available_extras": available_extras(),
        "hierarchy": {
            "max_depth": 32,
            "max_nodes": 256,
            "labels_pattern": r"^[A-Za-z0-9_-]{1,32}$",
            "modes": {
                "fanout_policy": ["uniform", "random", "explicit"],
                "sharing": ["independent", "shared_trunk", "tied_weights"],
                "delegation": ["broadcast", "select_k", "route_by_role", "confidence", "explicit"],
                "synthesis": ["consensus", "weighted_consensus", "best_by_score", "learned", "concat", "explicit"],
                "rollup": ["kappa_weighted", "mean", "minimum", "worst_case", "sum", "explicit"],
            },
        },
        "limits": {
            "preview_max_nodes": int(api.get("PREVIEW_MAX_NODES", 64)),
            "inference_max_workers": int(api.get("INFERENCE_MAX_WORKERS", 4)),
            "inference_budget_default": api.get("INFERENCE_BUDGET_DEFAULT"),
        },
    }


class ScorerListView(ApiView):
    summary = "Scorer stage types"
    response_serializer = ScorerCapabilitySerializer

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse(capabilities_payload(request)["scorer_types"])


class TrainerCapabilitiesView(ApiView):
    summary = "Optimizer, loss, scorer and hierarchy capabilities, and which extras are installed"
    response_serializer = TrainerCapabilitiesSerializer

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse(capabilities_payload(request))


class TrainerPresetListView(ApiView):
    summary = "Named trainer presets from settings"

    def get(self, request: Request) -> JsonResponse:
        presets = (deps.get_settings_dep(request).get("TRAINER_PRESETS") or {})
        return JsonResponse([{"name": name, "spec": dict(spec)} for name, spec in sorted(dict(presets).items())])


def _template_module(settings: Any) -> Any:
    from ....orchestration.orchestrator import _default_modules

    modules = _default_modules(settings)
    if not modules:
        raise ValidationError(
            {"trainer": ["no PEERS are configured, so there is no model to build the preview from."]}
        )
    return next(iter(modules.values()))


def build_preview(request: Request, spec: dict[str, Any]) -> dict[str, Any]:
    """Build the tree and optimizers without training; return the topology and costs."""
    from ....coopetition.tree import HierarchySpec
    from ....training.trainer import MACTFTrainer

    settings = deps.get_settings_dep(request)
    api = settings.get("API") or {}
    limit = int(api.get("PREVIEW_MAX_NODES", 64))
    if spec.get("hierarchy") is not None:
        planned = HierarchySpec.from_spec(spec["hierarchy"]).node_count()
    else:
        planned = int(spec.get("num_peers") or len(spec.get("peer_names") or ()) or len(settings.get("PEERS") or ()) or 1)
    if planned > limit:
        raise ValidationError(
            {"trainer": [
                f"the spec would build {planned} nodes but API.PREVIEW_MAX_NODES is {limit}; "
                "use GET /trainers/capabilities for the arithmetic instead of building the tree."
            ]}
        )

    kwargs = trainer_kwargs(spec, settings)
    template = _template_module(settings)
    if "hierarchy" not in kwargs:
        kwargs["model_factory"] = lambda name: copy.deepcopy(template)
        if "num_peers" not in kwargs and "peer_names" not in kwargs:
            kwargs["num_peers"] = max(1, len(settings.get("PEERS") or ()))
    else:
        import dataclasses

        kwargs["hierarchy"] = dataclasses.replace(kwargs["hierarchy"], root_models={"": template})
    trainer = None
    try:
        trainer = MACTFTrainer(**kwargs)
        trainer.build()
    except (ImproperlyConfigured, ContractError, MACTFError) as exc:
        raise ValidationError({"trainer": [str(exc)]}) from None
    try:
        nodes = []
        total = trainable_total = 0
        for node in trainer.tree.walk("root-first"):
            peer = node.peer
            params = list(peer.module.parameters()) if peer is not None else []
            count = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            total += count
            trainable_total += trainable
            nodes.append({"path": node.path, "parameters": count, "trainable": trainable, "frozen": count - trainable})
        claimed: set[int] = set()
        for optimizer in trainer.optimizers.values():
            claimed.update(id(p) for p in optimizer.parameters)
        unclaimed = 0
        for node in trainer.tree.walk("root-first"):
            if node.peer is None:
                continue
            unclaimed += sum(
                p.numel() for p in node.peer.module.parameters() if p.requires_grad and id(p) not in claimed
            )
        bytes_per_param = 8  # fp32 first and second moment: the Adam family's resident state
        tree = trainer.tree.to_dict()
        tree["source"] = "live"
        tree.setdefault("max_depth", int(spec.get("hierarchy", {}).get("max_depth", 8)) if spec.get("hierarchy") else 1)
        tree.setdefault("max_nodes", limit)
        return {
            "tree": tree,
            "nodes": nodes,
            "parameters": total,
            "trainable_parameters": trainable_total,
            "trainable_ratio": (trainable_total / total) if total else 0.0,
            "unclaimed_trainable_parameters": unclaimed,
            "all_trainable_claimed": unclaimed == 0,
            "projected_optimizer_bytes": trainable_total * bytes_per_param,
            "optimizers": {name: opt.describe() for name, opt in trainer.optimizers.items()},
            "available_extras": available_extras(),
            "spec": spec,
        }
    finally:
        if trainer is not None:
            trainer.close()


class TrainerPreviewView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Build a trainer's tree and optimizers without training"
    request_serializer = {"post": TrainerSpecSerializer}
    description = (
        "Answers 200 with the topology, per-node parameter counts, trainable ratio and projected "
        "optimizer memory, or 422 with the error the trainer would raise at build(). It never trains, "
        "writes a run row or allocates a checkpoint. ?stream=true sends the topology first over SSE."
    )

    def post(self, request: Request):
        spec = self.body(request, TrainerSpecSerializer)
        preview = build_preview(request, spec)
        if not bool_param(request, "stream"):
            return JsonResponse(preview)

        def stream():
            yield sse_event("topology", preview["tree"], event_id=1)
            yield sse_event("preview", preview, event_id=2)
            yield sse_event("terminal", {"status": "completed"}, event_id=3)

        return StreamingResponse(stream())
