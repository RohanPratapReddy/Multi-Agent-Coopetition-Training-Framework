"""PeerAdapter tests: any nn.Module, all output conventions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mactf.coopetition.adapter import PeerAdapter, infer_contract
from mactf.exceptions import ContractError


class TuplePeer(nn.Module):
    def __init__(self, state_dim=8, action_dim=3) -> None:
        super().__init__()
        self.body = nn.Linear(state_dim, 16)
        self.policy = nn.Linear(16, action_dim)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = torch.relu(self.body(state))
        return self.policy(hidden), self.value(hidden)


class DictPeer(nn.Module):
    def __init__(self, state_dim=8, action_dim=3) -> None:
        super().__init__()
        self.body = nn.Linear(state_dim, 16)
        self.policy = nn.Linear(16, action_dim)

    def forward(self, state):
        hidden = torch.relu(self.body(state))
        return {"logits": self.policy(hidden), "value": torch.zeros(state.shape[0], 1)}


class LogitsOnlyPeer(nn.Module):
    def __init__(self, state_dim=8, action_dim=3) -> None:
        super().__init__()
        self.body = nn.Linear(state_dim, 16)
        self.policy = nn.Linear(16, action_dim)

    def forward(self, state):
        return self.policy(torch.relu(self.body(state)))


class OneDimensionalPeer(nn.Module):
    """Returns a 1-D logits vector, which must be normalized to (1, action_dim)."""

    def __init__(self, action_dim=3) -> None:
        super().__init__()
        self.policy = nn.Linear(4, action_dim)

    def forward(self, state):
        return self.policy(state).squeeze(0)


@pytest.mark.parametrize(
    "peer_class", [TuplePeer, DictPeer, LogitsOnlyPeer, OneDimensionalPeer]
)
def test_adapter_normalizes_every_output_style(peer_class):
    module = peer_class()
    adapter = PeerAdapter(module, name="peer")
    state = torch.randn(5, 8) if hasattr(module, "body") and module.body.in_features == 8 else torch.randn(5, 4)
    logits, value = adapter(state)
    assert logits.ndim == 2
    assert value.ndim == 2
    assert value.shape[-1] == 1
    assert logits.shape[0] == value.shape[0]


def test_adapter_attaches_value_head_on_demand():
    module = LogitsOnlyPeer(state_dim=8, action_dim=3)
    adapter = PeerAdapter(module, name="peer")
    assert adapter.value_head is None
    logits, value = adapter(torch.randn(4, 8))
    assert adapter.value_head is not None
    assert value.shape == (4, 1)


def test_adapter_uses_native_value_head():
    module = TuplePeer()
    adapter = PeerAdapter(module, name="peer")
    _, value = adapter(torch.randn(4, 8))
    assert adapter.value_head is None
    assert adapter.contract.outputs_value is True


def test_adapter_export_import_roundtrip():
    adapter = PeerAdapter(TuplePeer(), name="peer")
    original = adapter.export_parameters()
    assert original.numel() > 0
    perturbed = original + 1.0
    adapter.import_parameters(perturbed)
    again = adapter.export_parameters()
    assert torch.allclose(again, perturbed, atol=1e-6)


def test_adapter_import_rejects_wrong_length():
    adapter = PeerAdapter(TuplePeer(), name="peer")
    with pytest.raises(ContractError):
        adapter.import_parameters(torch.zeros(3))


def test_adapter_rejects_non_module():
    with pytest.raises(ContractError):
        PeerAdapter(object())


def test_adapter_handles_frozen_parameters_in_aggregation():
    module = TuplePeer()
    for parameter in module.body.parameters():
        parameter.requires_grad_(False)
    adapter = PeerAdapter(module, name="peer")
    # Aggregation covers frozen parameters, training does not.
    assert len(list(adapter.aggregation_parameters())) > len(list(adapter.trainable_parameters()))
    assert adapter.export_parameters().numel() > 0


def test_contract_detects_attention_mask_support():
    class MaskedPeer(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return torch.zeros(1, 3)

    contract = infer_contract(MaskedPeer())
    assert contract.accepts_attention_mask
    assert contract.input_names == ("input_ids", "attention_mask")


def test_adapter_ignores_attention_mask_when_unsupported():
    adapter = PeerAdapter(LogitsOnlyPeer(), name="peer")
    logits, _ = adapter(torch.randn(2, 8), attention_mask=torch.ones(2))
    assert logits.shape == (2, 3)


def test_adapter_describe_is_informative():
    adapter = PeerAdapter(TuplePeer(), name="alpha")
    text = adapter.describe()
    assert "alpha" in text and "params=" in text


def test_value_strategy_none_yields_zeros():
    adapter = PeerAdapter(LogitsOnlyPeer(), name="peer", value_strategy="none")
    _, value = adapter(torch.randn(3, 8))
    assert torch.all(value == 0)
    assert adapter.value_head is None


def test_value_strategy_attach_requires_dims():
    with pytest.raises(ContractError):
        PeerAdapter(LogitsOnlyPeer(), name="peer", value_strategy="attach")
