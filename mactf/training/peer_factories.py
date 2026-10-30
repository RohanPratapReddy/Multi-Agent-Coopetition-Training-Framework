"""Built-in peer factories (document 17 section 12.3): clone, perturbed_clone, from_state."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..components.kinds import Origin
from ..components.registry import HUB


def clone(spec: Any, source: nn.Module | None = None) -> nn.Module:
    """A deep copy of ``source`` (the module of ``spec.like``)."""
    if source is None:
        raise ValueError("the 'clone' factory needs `like` (the node to copy)")
    return copy.deepcopy(source)


def perturbed_clone(spec: Any, source: nn.Module | None = None, *, std: float = 0.01) -> nn.Module:
    """A deep copy of ``source`` with Gaussian noise added to every floating-point parameter.

    The noise comes from the generator the engine has already seeded for this node
    (``node_seed(run_seed, path, topology_epoch)``), so the result is reproducible and the
    run's own random stream is untouched. A newcomer that starts identical to a sibling
    would receive identical gradients and add nothing to the swarm; a small perturbation
    gives it a different trajectory without throwing away what the sibling learned.
    """
    if source is None:
        raise ValueError("the 'perturbed_clone' factory needs `like` (the node to copy)")
    if isinstance(std, bool) or not isinstance(std, (int, float)) or std < 0 or std != std:
        raise ValueError("std must be a finite number >= 0")
    module = copy.deepcopy(source)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point() and parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * float(std))
    return module


def from_state(spec: Any, source: nn.Module | None = None, *, path: str, strict: bool = True) -> nn.Module:
    """A copy of ``source`` whose weights are read from a ``torch.save``-d state dict at ``path``."""
    if source is None:
        raise ValueError("the 'from_state' factory needs `like` (an architecture to load into)")
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"no state file at {file}")
    state = torch.load(file, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    module = copy.deepcopy(source)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(f"state file does not match the architecture: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return module


def register_builtins() -> None:
    """Register the three built-ins. Idempotent; called once at the end of this module."""
    registry = HUB.registry("peer_factory")
    for name, factory, description in (
        ("clone", clone, "deep copy of the `like` node"),
        ("perturbed_clone", perturbed_clone, "deep copy of the `like` node plus seeded Gaussian noise"),
        ("from_state", from_state, "the `like` architecture loaded from a state-dict file"),
    ):
        if registry.find(name) is None:
            registry.register(
                name, factory, origin=Origin.BUILTIN, description=description,
                capabilities={"finetuned": True, "needs_like": True},
            )


__all__ = ["clone", "from_state", "perturbed_clone", "register_builtins"]

register_builtins()
