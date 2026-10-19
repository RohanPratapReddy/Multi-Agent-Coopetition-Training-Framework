"""Small helpers shared by phases, kept out of the phase module to avoid cycles."""

from __future__ import annotations

from typing import Any

import torch


def state_targets_for(context: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the epoch's first task states and targets, moved to the runtime device."""
    episodes = getattr(context, "episodes", None)
    if not episodes:
        return None, None
    task = episodes[-1].task
    runtime = context.runtime
    states = runtime.to(task.states)
    targets = task.targets.to(states.device)
    return states, targets


def flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def describe_tensor(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    return f"tensor{tuple(tensor.shape)} on {tensor.device} ({tensor.dtype})"
