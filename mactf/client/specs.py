"""Builders for the ``trainer`` object of ``POST /runs`` and ``POST /trainers/preview``.

Each returns a plain ``dict`` in the exact shape the server's ``TrainerSpec`` serializer
accepts, so the result can be sent, stored, diffed and round-tripped without this module.
``None`` arguments are omitted rather than sent as ``null``, which keeps the payload
minimal and the server's own defaults in charge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["adamw", "consensus", "lora", "optimizer", "scorers", "trainer"]


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def optimizer(
    type: str = "torch",
    *,
    class_name: str | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    betas: Sequence[float] | None = None,
    method: str | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    grad_accumulation: int | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """An ``OptimizerSpec``. ``type`` is ``torch``, ``scipy``, ``callable`` or ``router``."""
    return _drop_none({
        "type": type, "class_name": class_name, "lr": lr, "weight_decay": weight_decay,
        "momentum": momentum, "betas": list(betas) if betas is not None else None, "method": method,
        "max_iter": max_iter, "tol": tol, "grad_accumulation": grad_accumulation,
        "max_grad_norm": max_grad_norm,
    })


def adamw(lr: float = 1e-4, *, weight_decay: float = 0.01, betas: Sequence[float] = (0.9, 0.999), **extra: Any) -> dict[str, Any]:
    """The torch ``AdamW`` optimizer (the server names optimizers by class, ``type`` is ``torch``)."""
    return optimizer("torch", class_name="AdamW", lr=lr, weight_decay=weight_decay, betas=betas, **extra)


def lora(
    r: int = 8,
    *,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] | str | None = None,
    peft_type: str = "lora",
    quantization: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A ``FinetuneSpec`` for LoRA (``quantization="4bit"`` gives QLoRA)."""
    modules: Any = list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules
    return _drop_none({
        "peft_type": peft_type, "r": r, "alpha": alpha, "dropout": dropout,
        "target_modules": modules, "quantization": quantization, **extra,
    })


def scorers(*chain: Mapping[str, Any] | str, severity_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """A ``ScorerChainSpec``; a bare string is a scorer ``type`` with no config."""
    entries = [{"type": item} if isinstance(item, str) else dict(item) for item in chain]
    return _drop_none({"scorers": entries, "severity_weights": dict(severity_weights) if severity_weights else None})


def consensus(
    *,
    levels: Sequence[Mapping[str, Any] | tuple[str, int]] = (),
    synthesis: str = "consensus",
    delegation: str = "broadcast",
    roots: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    """A ``HierarchySpec`` whose parents synthesise their children's answers by consensus.

    ``levels`` may be ``(label, count)`` pairs or full level dicts.
    """
    resolved = [
        {"label": item[0], "count": item[1]} if isinstance(item, tuple) else dict(item) for item in levels
    ]
    return {"levels": resolved, "roots": roots, "synthesis": synthesis, "delegation": delegation, **extra}


def trainer(
    *,
    num_peers: int | None = None,
    peer_names: Sequence[str] | None = None,
    hierarchy: Mapping[str, Any] | None = None,
    optimizer: Mapping[str, Any] | None = None,
    loss: Mapping[str, Any] | None = None,
    finetuning: Mapping[str, Any] | None = None,
    scorers: Mapping[str, Any] | None = None,
    peer_strategy: Mapping[str, Any] | None = None,
    weight_update: Mapping[str, Any] | None = None,
    grad_accumulation: int | None = None,
    max_grad_norm: float | None = None,
    steps_per_epoch: int | None = None,
    distillation: Mapping[str, Any] | bool | None = None,
) -> dict[str, Any]:
    """A ``TrainerSpec``. ``hierarchy`` and ``num_peers`` are mutually exclusive, as on the server.

    ``distillation`` takes the same polymorphic value as ``MACTFTrainer`` (a mapping, a bare
    ``True`` or ``False``, or ``None`` to defer to the project's settings). A bare boolean is
    normalised to ``{"ENABLED": <bool>}`` because the wire field is a mapping shape.
    """
    if hierarchy is not None and num_peers is not None:
        raise ValueError("hierarchy and num_peers are mutually exclusive")
    distillation_block = (
        None
        if distillation is None
        else dict(distillation)
        if isinstance(distillation, Mapping)
        else {"ENABLED": bool(distillation)}
    )
    return _drop_none({
        "num_peers": num_peers,
        "peer_names": list(peer_names) if peer_names is not None else None,
        "hierarchy": dict(hierarchy) if hierarchy is not None else None,
        "optimizer": dict(optimizer) if optimizer is not None else None,
        "loss": dict(loss) if loss is not None else None,
        "finetuning": dict(finetuning) if finetuning is not None else None,
        "scorers": dict(scorers) if scorers is not None else None,
        "peer_strategy": dict(peer_strategy) if peer_strategy is not None else None,
        "weight_update": dict(weight_update) if weight_update is not None else None,
        "grad_accumulation": grad_accumulation,
        "max_grad_norm": max_grad_norm,
        "steps_per_epoch": steps_per_epoch,
        "distillation": distillation_block,
    })
