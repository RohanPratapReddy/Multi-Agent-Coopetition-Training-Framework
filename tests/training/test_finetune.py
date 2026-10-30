"""Criterion 19 (adapter half): LoRA trains only adapters; nothing is silently dropped."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mactf.exceptions import ImproperlyConfigured, ValidationError
from mactf.training.finetune import (
    AUTO_TARGET_MODULES,
    FinetuneConfig,
    adapter_state,
    apply_finetuning,
    merge_adapters,
    resolve_peft,
    resolve_target_modules,
    trainable_report,
)

from train_support import Tiny


def _peft_importable() -> bool:
    try:
        import peft  # noqa: F401
    except Exception:  # an ImportError, or a broken native dependency (torchaudio DLL)
        return False
    return True


needs_peft = pytest.mark.skipif(not _peft_importable(), reason="peft is not importable here")


def test_config_validation():
    for kwargs in (
        {"peft_type": "mystery"},
        {"peft_type": "lora", "r": 0},
        {"peft_type": "lora", "alpha": 0},
        {"peft_type": "lora", "dropout": 1.0},
        {"peft_type": "lora", "bias": "some"},
        {"peft_type": "lora", "quantization": "2bit"},
        {"peft_type": "lora", "compute_dtype": "int4"},
    ):
        with pytest.raises(ValidationError):
            FinetuneConfig(**kwargs)
    with pytest.raises(ImproperlyConfigured, match="4-bit"):
        FinetuneConfig(peft_type="qlora", quantization="8bit")


def test_qlora_implies_four_bit_and_single_target_is_normalised():
    config = FinetuneConfig(peft_type="qlora", target_modules="q_proj")
    assert config.effective_quantization == "4bit" and config.quantized
    assert config.target_modules == ("q_proj",)
    assert FinetuneConfig(peft_type="lora").quantized is False
    assert FinetuneConfig().enabled is False


def test_from_spec_semantics():
    assert FinetuneConfig.from_spec(None) is None
    assert FinetuneConfig.from_spec({}) is None
    config = FinetuneConfig.from_spec({"peft_type": "lora", "target_modules": ["a", "b"]})
    assert config.target_modules == ("a", "b")
    assert FinetuneConfig.from_spec(config) is config
    with pytest.raises(ValidationError, match="does not accept 'bogus'"):
        FinetuneConfig.from_spec({"peft_type": "lora", "bogus": 1})
    with pytest.raises(ValidationError):
        FinetuneConfig.from_spec(5)
    assert FinetuneConfig.from_spec(config.as_dict()).as_dict() == config.as_dict()


def test_no_finetuning_returns_the_model_untouched():
    model = Tiny()
    same, report = apply_finetuning(model, None)
    assert same is model and report == {}


def test_missing_peft_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImproperlyConfigured, match=r'pip install "mactf\[peft\]"'):
        resolve_peft()
    assert resolve_peft(required=False) is None
    with pytest.raises(ImproperlyConfigured, match=r'mactf\[peft\]'):
        apply_finetuning(Tiny(), {"peft_type": "lora", "target_modules": ["policy"]})


def test_unknown_architecture_lists_the_linear_modules_it_found():
    with pytest.raises(ImproperlyConfigured) as raised:
        resolve_target_modules(Tiny(), FinetuneConfig(peft_type="lora"))
    message = str(raised.value)
    assert "unknown architecture" in message
    assert "policy" in message and "value" in message


def test_known_architecture_keeps_only_modules_that_exist():
    family, names = next(iter(AUTO_TARGET_MODULES.items()))
    model = nn.Module()
    model.config = SimpleNamespace(model_type=family)
    keep = names[0]
    setattr(model, "wrapper", nn.Sequential())
    model.wrapper.add_module(keep, nn.Linear(2, 2))
    resolved = resolve_target_modules(model, FinetuneConfig(peft_type="lora"))
    assert resolved == (keep,)
    bare = nn.Module()
    bare.config = SimpleNamespace(model_type=family)
    with pytest.raises(ImproperlyConfigured, match="none"):
        resolve_target_modules(bare, FinetuneConfig(peft_type="lora"))
    explicit = resolve_target_modules(
        Tiny(), FinetuneConfig(peft_type="lora", target_modules=("policy",))
    )
    assert explicit == ("policy",)
    assert (
        resolve_target_modules(Tiny(), FinetuneConfig(peft_type="lora", target_modules="all-linear"))
        == "all-linear"
    )


def test_trainable_report_counts_frozen_and_adapters():
    model = Tiny()
    for parameter in model.body.parameters():
        parameter.requires_grad_(False)
    report = trainable_report(model, adapter_names=("default",))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert report["total_parameters"] == total
    assert report["trainable_parameters"] == trainable
    assert report["trainable_ratio"] == pytest.approx(trainable / total)
    assert report["frozen_base"] is True and report["adapter_names"] == ["default"]


def test_adapter_state_and_merge_need_a_peft_model():
    assert adapter_state(Tiny()) == {}
    with pytest.raises(ImproperlyConfigured, match="merge_and_unload"):
        merge_adapters(Tiny())


@needs_peft
def test_lora_trains_only_the_adapters():
    model = Tiny()
    wrapped, report = apply_finetuning(
        model, {"peft_type": "lora", "r": 2, "alpha": 4, "target_modules": ["policy", "value"]}
    )
    trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert 0.0 < report["trainable_ratio"] < 1.0
    assert report["peft_type"] == "lora"
    again, repeat = apply_finetuning(wrapped, {"peft_type": "lora", "target_modules": ["policy"]})
    assert again is wrapped
    assert repeat["trainable_parameters"] == report["trainable_parameters"]
    state = adapter_state(wrapped)
    assert state and all("lora_" in key for key in state)
    out = wrapped(torch.randn(3, 4))
    assert out[0].shape == (3, 3)
