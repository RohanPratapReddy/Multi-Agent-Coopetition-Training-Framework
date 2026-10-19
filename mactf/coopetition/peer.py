"""The peer runtime: wrapping a module plus its optimizer, scheduler, and identity."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from ..exceptions import ContractError
from ..runtime import Runtime
from .adapter import PeerAdapter


@dataclass
class Peer:
    """A trainable peer in the swarm.

    ``module`` is the user's ``nn.Module``; ``adapter`` is the framework wrapper
    that the optimizer and aggregator act on. Because the adapter holds the
    module by reference, both refer to the same parameters.
    """

    name: str
    adapter: PeerAdapter
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> nn.Module:
        return self.adapter.module

    @property
    def parameters(self) -> Iterable[nn.Parameter]:
        return self.adapter.parameters()

    def to(self, runtime: Runtime) -> "Peer":
        self.adapter = runtime.to(self.adapter)
        if self.optimizer is not None:
            # Rebuild the optimizer so it points at the moved parameters rather
            # than stale CPU tensors.
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr)
        return self

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        if self.optimizer is not None:
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.step()

    def export_parameters(self) -> torch.Tensor:
        return self.adapter.export_parameters()

    def import_parameters(self, flat: torch.Tensor) -> None:
        self.adapter.import_parameters(flat)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "metadata": dict(self.metadata),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.adapter.load_state_dict(payload["adapter"])
        if self.optimizer is not None and payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.metadata.update(payload.get("metadata", {}))

    def describe(self) -> str:
        return f"Peer({self.name}, {self.adapter.describe()})"


def build_peer(
    module: nn.Module,
    *,
    name: str | None = None,
    runtime: Runtime,
    learning_rate: float = 1e-3,
    action_dim: int | None = None,
    state_dim: int | None = None,
    weight_decay: float = 0.0,
    value_strategy: str = "auto",
) -> Peer:
    """Wrap a module into a :class:`Peer` with an Adam optimizer."""
    adapter = PeerAdapter(
        module,
        name=name,
        action_dim=action_dim,
        state_dim=state_dim,
        value_strategy=value_strategy,  # type: ignore[arg-type]
    )
    adapter = runtime.to(adapter)
    optimizer = torch.optim.Adam(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return Peer(
        name=name or adapter.peer_name,
        adapter=adapter,
        optimizer=optimizer,
    )


def build_reference(peer: Peer, *, runtime: Runtime, name: str | None = None) -> Peer:
    """Deep-copy a peer into a frozen reference policy (framework.md 11.5)."""
    adapter = copy.deepcopy(peer.adapter)
    adapter = runtime.to(adapter)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return Peer(
        name=name or f"{peer.name}__reference",
        adapter=adapter,
        optimizer=None,
        metadata={"frozen": True},
    )


def clone_peer(peer: Peer, *, runtime: Runtime, name: str) -> Peer:
    """Create an independent peer initialized from another's weights."""
    module = copy.deepcopy(peer.module)
    return build_peer(
        module,
        name=name,
        runtime=runtime,
        learning_rate=(
            peer.optimizer.param_groups[0]["lr"] if peer.optimizer is not None else 1e-3
        ),
        value_strategy=peer.adapter.value_strategy,
    )


def require_same_layout(peers: dict[str, Peer]) -> None:
    """Fail early when peers cannot be aggregated because of layout mismatch."""
    shapes = {name: peer.adapter.parameter_shapes() for name, peer in peers.items()}
    if not shapes:
        return
    reference_name, reference_shapes = next(iter(shapes.items()))
    for name, candidate in shapes.items():
        if candidate != reference_shapes:
            raise ContractError(
                f"peer {name!r} has a different parameter layout than {reference_name!r}; "
                "reward-modulated aggregation requires identical architectures"
            )
