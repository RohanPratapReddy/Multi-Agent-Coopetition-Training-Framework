"""Wrapping arbitrary ``torch.nn.Module`` subclasses as framework peers.

The framework must accept *any* module, so it does not assume an output shape.
:class:`PeerAdapter` normalizes the common conventions:

* ``(logits, value)`` tuple
* ``{"logits": ..., "value": ...}`` mapping
* ``logits`` tensor only, in which case a value head is synthesized
* a ``ModelOutput``-like object with ``logits``/``last_hidden_state`` attributes

It also exposes a parameter API for aggregation: :meth:`PeerAdapter.export_parameters`
returns a flat float tensor and :meth:`PeerAdapter.import_parameters` writes it
back, which is what the coopetition layer averages.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn

from ..exceptions import ContractError

ValueStrategy = Literal["auto", "attach", "none"]


@dataclass
class PeerContract:
    """The resolved input/output contract for a peer module."""

    input_names: tuple[str, ...] = ("state",)
    accepts_attention_mask: bool = False
    outputs_value: bool = False
    output_kind: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"inputs={self.input_names}", f"output={self.output_kind}"]
        if self.outputs_value:
            parts.append("value=native")
        else:
            parts.append("value=synthesized")
        return ", ".join(parts)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def infer_contract(module: nn.Module) -> PeerContract:
    """Inspect a module's signature to describe how it expects to be called."""
    contract = PeerContract()
    try:
        signature = inspect.signature(module.forward)
        names = [
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY, parameter.KEYWORD_ONLY}
        ]
        contract.input_names = tuple(names) or ("state",)
        contract.accepts_attention_mask = "attention_mask" in names
    except (TypeError, ValueError):  # pragma: no cover - C++/scripted modules
        contract.notes.append("forward signature not introspectable; assuming (state,)")
    return contract


class PeerAdapter(nn.Module):
    """A framework-facing wrapper around an arbitrary ``nn.Module``.

    Parameters
    ----------
    module:
        Any ``nn.Module``. Its parameters are *not* copied; the adapter holds the
        same object, so optimizers built on the adapter train the original.
    action_dim:
        Required when the module returns logits only, so a value head can be
        attached. If omitted, the adapter infers it from the first forward call.
    value_strategy:
        ``auto`` attaches a value head only when the module does not return one;
        ``attach`` always attaches; ``none`` synthesizes zeros and never trains a
        value head (valid for pure policy-gradient runs).
    state_dim:
        Hidden size hint for the attached value head. Inferred from the logits
        when possible.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        name: str | None = None,
        action_dim: int | None = None,
        state_dim: int | None = None,
        value_strategy: ValueStrategy = "auto",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise ContractError(f"expected an nn.Module, got {type(module)!r}")
        if value_strategy not in {"auto", "attach", "none"}:
            raise ContractError(f"unknown value_strategy {value_strategy!r}")

        self.module = module
        self.peer_name = name or type(module).__name__
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.value_strategy = value_strategy
        self.contract = infer_contract(module)
        self.value_head: nn.Module | None = None
        self._contract_resolved = value_strategy == "none"
        self._last_hidden: torch.Tensor | None = None

        if freeze_backbone:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)
            self.contract.notes.append("backbone frozen by request")

        if value_strategy == "attach":
            if action_dim is None or state_dim is None:
                raise ContractError(
                    "value_strategy='attach' requires both action_dim and state_dim"
                )
            self._attach_value_head(state_dim)

    # -- value head management -------------------------------------------
    def _attach_value_head(self, hidden_dim: int) -> None:
        if self.value_head is not None:
            return
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )
        self.contract.outputs_value = False
        self._contract_resolved = True
        self.contract.notes.append(f"attached value head over hidden dim {hidden_dim}")

    def _resolve_contract(self, output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split a module output into ``(logits, value_or_None)``."""
        if isinstance(output, dict):
            logits = output.get("logits", output.get("policy_logits", output.get("action_logits")))
            value = output.get("value", output.get("values", output.get("v")))
            if logits is None:
                tensor = _first_tensor(list(output.values()))
                if tensor is None:
                    raise ContractError(
                        f"{self.peer_name}: dict output has no tensor to treat as logits"
                    )
                logits = tensor
            return logits, value

        if isinstance(output, (tuple, list)):
            if not output:
                raise ContractError(f"{self.peer_name}: forward returned an empty {type(output).__name__}")
            logits = output[0]
            value = output[1] if len(output) > 1 else None
            if not torch.is_tensor(logits):
                tensor = _first_tensor(output)
                if tensor is None:
                    raise ContractError(
                        f"{self.peer_name}: tuple output contains no tensor"
                    )
                logits = tensor
            return logits, value

        if torch.is_tensor(output):
            return output, None

        # Attribute-style outputs (transformers' ModelOutput and friends).
        logits = getattr(output, "logits", None) or getattr(output, "policy_logits", None)
        value = getattr(output, "value", None) or getattr(output, "values", None)
        if logits is not None:
            return logits, value
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            return hidden, value
        tensor = _first_tensor(output)
        if tensor is not None:
            return tensor, None
        raise ContractError(
            f"{self.peer_name}: cannot interpret {type(output)!r} as logits; "
            f"return a tensor, a (logits, value) tuple, or a dict with 'logits'"
        )

    def _ensure_value(self, logits: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        """Return a value estimate, attaching a head when the module lacks one."""
        if self.value_strategy == "none":
            return torch.zeros(logits.shape[:-1] + (1,), device=logits.device, dtype=logits.dtype)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if self.value_head is None:
            source = hidden if hidden is not None else logits
            hidden_dim = int(source.shape[-1])
            # The head lives on the adapter so it is trained by the same optimizer
            # and participates in parameter aggregation.
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.Tanh(),
                nn.Linear(max(8, hidden_dim // 2), 1),
            ).to(device=source.device, dtype=source.dtype)
            self.contract.notes.append(
                f"attached value head at first forward, hidden dim {hidden_dim}"
            )
        source = hidden if hidden is not None else logits
        if source.shape[-1] != self.value_head[0].in_features:
            raise ContractError(
                f"{self.peer_name}: value head expects {self.value_head[0].in_features} "
                f"features but the module produced {source.shape[-1]}; pass state_dim explicitly"
            )
        return self.value_head(source)

    # -- forward ---------------------------------------------------------
    def forward(self, state: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the module and return ``(logits, value)`` with ``value`` shaped (..., 1)."""
        call_kwargs = dict(kwargs)
        if not self.contract.accepts_attention_mask:
            call_kwargs.pop("attention_mask", None)

        output = self.module(state, **call_kwargs)
        logits, native_value = self._resolve_contract(output)

        if not torch.is_tensor(logits):
            raise ContractError(f"{self.peer_name}: logits must be a tensor")
        if logits.ndim == 0:
            logits = logits.reshape(1, 1)
        elif logits.ndim == 1:
            logits = logits.unsqueeze(0)

        hidden = None
        if isinstance(output, (tuple, list)) and len(output) > 2 and torch.is_tensor(output[2]):
            hidden = output[2]

        if native_value is not None and torch.is_tensor(native_value):
            value = native_value
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            while value.ndim < logits.ndim:
                value = value.unsqueeze(-1)
            self.contract.outputs_value = True
            return logits, value

        return logits, self._ensure_value(logits, hidden)

    # -- parameter API for aggregation ------------------------------------
    def trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def aggregation_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Every parameter, frozen or not, in a stable order.

        Aggregation must cover the whole vector even when a peer has frozen layers,
        otherwise blending a trained peer with the frozen reference would produce
        a length mismatch instead of a meaningful average.
        """
        return list(self.named_parameters())

    def export_parameters(self) -> torch.Tensor:
        """Flatten every parameter into one float tensor (CPU, float32)."""
        chunks = []
        for _, parameter in self.aggregation_parameters():
            chunks.append(parameter.detach().reshape(-1).to(device="cpu", dtype=torch.float32))
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks)

    @torch.no_grad()
    def import_parameters(self, flat: torch.Tensor, *, strict: bool = True) -> int:
        """Write a flat tensor back into the module's parameters."""
        expected = sum(parameter.numel() for _, parameter in self.aggregation_parameters())
        if strict and flat.numel() != expected:
            raise ContractError(
                f"{self.peer_name}: parameter vector has {flat.numel()} values, expected {expected}"
            )
        offset = 0
        for _, parameter in self.aggregation_parameters():
            count = parameter.numel()
            if offset + count > flat.numel():
                break
            chunk = flat[offset : offset + count].to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(chunk.reshape_as(parameter))
            offset += count
        return offset

    def parameter_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        return [(name, tuple(parameter.shape)) for name, parameter in self.aggregation_parameters()]

    def describe(self) -> str:
        total = sum(parameter.numel() for _, parameter in self.trainable_parameters())
        return (
            f"PeerAdapter({self.peer_name}, params={total}, contract=[{self.contract.describe()}])"
        )
