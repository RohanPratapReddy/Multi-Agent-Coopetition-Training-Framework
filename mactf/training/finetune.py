"""Parameter-efficient fine-tuning (document 08 section 4).

``apply_finetuning`` is the one entry point. It is deliberately strict about the
three things that silently produce a mislabelled run:

* QLoRA without CUDA is an error, not a silent fallback to fp32. A user who asked
  for 4-bit to fit memory needs to know it did not happen.
* ``target_modules="auto"`` resolves by inspecting ``model.config.model_type``
  against a documented table, and an unknown architecture raises with the list of
  names the caller could target instead of guessing ``q_proj``.
* The base is frozen *before* PEFT attaches adapters when ``freeze_base=True``, so
  the optimizer built afterwards sees only adapter parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch.nn as nn

from ..exceptions import ImproperlyConfigured, ValidationError

__all__ = [
    "AUTO_TARGET_MODULES",
    "FinetuneConfig",
    "apply_finetuning",
    "resolve_peft",
    "resolve_target_modules",
    "trainable_report",
]

PeftType = Literal["lora", "adalora", "prefix", "qlora"]

#: Per-architecture attention projection names for ``target_modules="auto"``.
#:
#: Keys are ``transformers`` ``model_type`` values. Values are the module suffixes
#: the architecture's attention blocks actually use; a wrong name here fails
#: loudly at wrap time (PEFT raises on an empty target set) rather than silently
#: training nothing.
AUTO_TARGET_MODULES: dict[str, tuple[str, ...]] = {
    # Llama-family decoder attention.
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mistral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "qwen2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "qwen2_moe": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "falcon": ("query_key_value", "dense"),
    "mixtral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gemma": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gemma2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "phi": ("q_proj", "k_proj", "v_proj", "dense"),
    "phi3": ("qkv_proj", "o_proj"),
    "gpt2": ("c_attn", "c_proj"),
    "gpt_neox": ("query_key_value", "dense"),
    "bloom": ("query_key_value", "dense"),
    "opt": ("q_proj", "k_proj", "v_proj", "out_proj"),
    # Encoder attention.
    "bert": ("query", "key", "value"),
    "roberta": ("query", "key", "value"),
    "distilbert": ("q_lin", "k_lin", "v_lin", "out_lin"),
    "electra": ("query", "key", "value"),
    "deberta": ("query_proj", "key_proj", "value_proj"),
    "deberta-v2": ("query_proj", "key_proj", "value_proj"),
    # Vision / multimodal, where "attention" is often fused.
    "vit": ("query", "key", "value"),
    "t5": ("q", "k", "v", "o"),
}


@dataclass(frozen=True)
class FinetuneConfig:
    """Validated fine-tuning configuration.

    A frozen dataclass, so it is safe to share across nodes: the trainer applies
    one config to every node's fresh module.
    """

    peft_type: PeftType | None = None
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] | str = "auto"
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    quantization: Literal["4bit", "8bit"] | None = None
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    freeze_base: bool = True
    gradient_checkpointing: bool = False
    modules_to_save: tuple[str, ...] = ()
    init_lora_weights: str = "default"

    def __post_init__(self) -> None:
        if self.peft_type not in (None, "lora", "adalora", "prefix", "qlora"):
            raise ValidationError(
                f"unknown peft_type {self.peft_type!r}; use 'lora', 'adalora', "
                "'prefix', 'qlora', or None"
            )
        if self.r < 1:
            raise ValidationError(f"FinetuneConfig.r must be >= 1, got {self.r!r}")
        if self.alpha < 1:
            raise ValidationError(f"FinetuneConfig.alpha must be >= 1, got {self.alpha!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(
                f"FinetuneConfig.dropout must be in [0, 1), got {self.dropout!r}"
            )
        if self.bias not in ("none", "all", "lora_only"):
            raise ValidationError(
                f"unknown bias {self.bias!r}; use 'none', 'all', or 'lora_only'"
            )
        if self.quantization not in (None, "4bit", "8bit"):
            raise ValidationError(
                f"unknown quantization {self.quantization!r}; use '4bit', '8bit', or None"
            )
        if self.compute_dtype not in ("bfloat16", "float16", "float32"):
            raise ValidationError(
                f"unknown compute_dtype {self.compute_dtype!r}; use 'bfloat16', "
                "'float16', or 'float32'"
            )
        # QLoRA *is* 4-bit LoRA. Requesting it with 8-bit is a contradiction, and
        # accepting it would produce a run labelled QLoRA that used half the
        # memory saving the user asked for.
        if self.peft_type == "qlora" and self.quantization not in (None, "4bit"):
            raise ImproperlyConfigured(
                f"peft_type='qlora' implies 4-bit quantization, but quantization="
                f"{self.quantization!r} was requested; either drop quantization or use "
                "peft_type='lora'"
            )
        if isinstance(self.target_modules, list):
            object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if isinstance(self.target_modules, str) and self.target_modules not in (
            "auto",
            "all-linear",
        ):
            # A single module name is a legitimate convenience.
            object.__setattr__(self, "target_modules", (self.target_modules,))

    # -- derived ---------------------------------------------------------
    @property
    def effective_quantization(self) -> str | None:
        """``qlora`` resolves to ``4bit`` even when ``quantization`` is unset."""
        if self.quantization is not None:
            return self.quantization
        return "4bit" if self.peft_type == "qlora" else None

    @property
    def quantized(self) -> bool:
        return self.effective_quantization is not None

    @property
    def enabled(self) -> bool:
        return self.peft_type is not None or self.quantized

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "peft_type": self.peft_type,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "quantization": self.quantization,
            "effective_quantization": self.effective_quantization,
            "quant_type": self.quant_type,
            "double_quant": self.double_quant,
            "compute_dtype": self.compute_dtype,
            "freeze_base": self.freeze_base,
            "gradient_checkpointing": self.gradient_checkpointing,
            "modules_to_save": list(self.modules_to_save),
            "init_lora_weights": self.init_lora_weights,
        }
        payload["target_modules"] = (
            list(self.target_modules)
            if isinstance(self.target_modules, tuple)
            else self.target_modules
        )
        return payload

    @classmethod
    def from_spec(cls, spec: "FinetuneConfig | Mapping[str, Any] | None") -> "FinetuneConfig | None":
        if spec is None:
            return None
        if isinstance(spec, FinetuneConfig):
            return spec
        if not isinstance(spec, Mapping):
            raise ValidationError(
                f"finetuning must be a FinetuneConfig or a mapping, got {type(spec).__name__}"
            )
        payload = dict(spec)
        payload.pop("effective_quantization", None)  # derived, emitted by as_dict()
        if not payload or payload.get("peft_type") is None and not payload.get("quantization"):
            # An empty mapping means "no fine-tuning", which is not an error.
            return None
        accepted = set(FinetuneConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - accepted)
        if unknown:
            raise ValidationError(
                "FinetuneConfig does not accept "
                + ", ".join(repr(key) for key in unknown)
                + "; accepted keys: "
                + ", ".join(sorted(accepted))
            )
        target = payload.get("target_modules", "auto")
        if isinstance(target, list):
            payload["target_modules"] = tuple(str(name) for name in target)
        payload["modules_to_save"] = tuple(payload.get("modules_to_save", ()) or ())
        return cls(**payload)


def resolve_peft(*, required: bool = True) -> Any:
    """Import ``peft``, raising the extra's pip command when it is missing."""
    try:
        import peft  # noqa: PLC0415 - optional dependency by design
    except ImportError as exc:
        if not required:
            return None
        raise ImproperlyConfigured(
            "fine-tuning requires the 'peft' extra; install it with "
            '`pip install "mactf[peft]"` (peft>=0.10, transformers>=4.40)'
        ) from exc
    return peft


def resolve_target_modules(
    model: nn.Module,
    config: FinetuneConfig,
) -> tuple[str, ...] | str:
    """Resolve ``target_modules`` for PEFT, inspecting the model when ``auto``.

    An unknown architecture raises with the parameter names present in the model,
    because the alternative — defaulting to ``q_proj`` — attaches adapters to
    nothing and reports a trainable ratio of zero three epochs later.
    """
    target = config.target_modules
    if target not in ("auto", "all-linear"):
        return tuple(target) if isinstance(target, (tuple, list)) else target

    if target == "all-linear":
        return "all-linear"

    model_type = ""
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_type = str(getattr(model_config, "model_type", "") or "")
    if not model_type:
        # A submodule of a transformers model often lacks its own config.
        for attribute in ("base_model", "model"):
            nested = getattr(model, attribute, None)
            nested_config = getattr(nested, "config", None) if nested is not None else None
            if nested_config is not None:
                model_type = str(getattr(nested_config, "model_type", "") or "")
                if model_type:
                    break

    table = AUTO_TARGET_MODULES.get(model_type)
    if table is not None:
        # Only keep names the model actually has, so a hybrid architecture that
        # renamed one projection does not make PEFT raise.
        present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        selected = tuple(name for name in table if name in present)
        if selected:
            return selected
        raise ImproperlyConfigured(
            f"model_type {model_type!r} maps to {list(table)}, but the model has none "
            "of those modules; pass target_modules explicitly"
        )

    candidates = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name
        }
    )
    raise ImproperlyConfigured(
        "cannot resolve target_modules='auto' for an unknown architecture "
        f"(model_type={model_type or 'absent'}); pass target_modules explicitly. "
        f"Linear modules present: {', '.join(candidates[:40]) or '(none)'}"
    )


def trainable_report(
    model: nn.Module,
    *,
    adapter_names: tuple[str, ...] = (),
    quantized: bool = False,
) -> dict[str, Any]:
    """Count trainable versus total parameters, and list adapter names.

    Named adapters are found by suffix, which is how PEFT names its injected
    modules (``lora_A``, ``lora_B``, ``lora_embedding_A``, ``prefix_encoder``).
    """
    total = 0
    trainable = 0
    for _name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": (trainable / total) if total else 0.0,
        "adapter_names": list(adapter_names),
        "quantized": bool(quantized),
        "frozen_base": trainable < total,
    }


def _quantization_config(config: FinetuneConfig, *, device: Any) -> Any:
    """Build a ``BitsAndBytesConfig``, refusing non-CUDA devices explicitly."""
    from ..runtime import resolve_device  # local import: runtime is a leaf module

    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise ImproperlyConfigured(
            "4-bit quantization requires CUDA and bitsandbytes; set "
            "FINETUNE.QUANTIZATION=None or use peft_type='lora'"
        )
    try:
        import bitsandbytes  # noqa: F401,PLC0415 - presence check for the extra
    except ImportError as exc:
        raise ImproperlyConfigured(
            "quantization requires the 'quant' extra; install it with "
            '`pip install "mactf[quant]"` (bitsandbytes>=0.43)'
        ) from exc
    try:
        import transformers  # noqa: PLC0415 - optional dependency by design
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImproperlyConfigured(
            "quantization requires the 'peft' extra; install it with "
            '`pip install "mactf[peft]"`'
        ) from exc

    dtype = {
        "bfloat16": "bfloat16",
        "float16": "float16",
        "float32": "float32",
    }[config.compute_dtype]
    return transformers.BitsAndBytesConfig(
        load_in_4bit=config.effective_quantization == "4bit",
        load_in_8bit=config.effective_quantization == "8bit",
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_compute_dtype=getattr(__import__("torch"), dtype),
    )


def _already_applied(model: nn.Module, config: FinetuneConfig) -> bool:
    """Whether ``model`` is already wrapped with a compatible PEFT config.

    Idempotence is compared on the fields that change behaviour, not on object
    identity: re-applying the same config must be a no-op, and applying a
    *different* config to an already-wrapped model is a bug worth raising.
    """
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping):
        return False
    existing = next(iter(peft_config.values()), None)
    if existing is None:
        return False
    type_name = type(existing).__name__.lower()
    expected = {
        "lora": "lora",
        "adalora": "adalora",
        "prefix": "prefix",
        "qlora": "lora",
    }.get(str(config.peft_type), "")
    if expected and expected not in type_name:
        raise ImproperlyConfigured(
            f"this model is already wrapped with {type(existing).__name__}, but "
            f"peft_type={config.peft_type!r} was requested; PEFT models cannot be "
            "re-wrapped with a different adapter type"
        )
    for attribute, requested in (
        ("r", config.r),
        ("lora_alpha", config.alpha),
        ("lora_dropout", config.dropout),
    ):
        current = getattr(existing, attribute, None)
        if current is not None and isinstance(requested, int) and isinstance(current, int):
            if int(current) != int(requested):
                raise ImproperlyConfigured(
                    f"this model is already wrapped with {attribute}={current}, but "
                    f"{attribute}={requested} was requested; load a fresh model or "
                    "reuse the original config"
                )
    return True


def apply_finetuning(
    model: nn.Module,
    config: "FinetuneConfig | Mapping[str, Any] | None",
    *,
    device: Any = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Wrap ``model`` for PEFT and return ``(model, report)``.

    Returns the model itself and an empty report when nothing is requested, so a
    caller can pass the result straight into an optimizer without a branch. The
    base is frozen before wrapping when ``freeze_base`` is set, which is what makes
    the subsequent optimizer bind see only adapter parameters.
    """
    resolved = FinetuneConfig.from_spec(config)
    if resolved is None or not resolved.enabled:
        return model, {}

    # Quantization is checked before the PEFT import so a CPU user gets the
    # actionable message rather than an ImportError for bitsandbytes.
    if resolved.quantized and resolved.effective_quantization is not None:
        _quantization_config(resolved, device=device)

    peft = resolve_peft()
    if _already_applied(model, resolved):
        return model, trainable_report(
            model,
            adapter_names=tuple(str(name) for name in getattr(model, "peft_config", {})),
            quantized=resolved.quantized,
        )

    if resolved.freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    if resolved.gradient_checkpointing:
        enabler = getattr(model, "gradient_checkpointing_enable", None)
        if enabler is None:
            raise ImproperlyConfigured(
                "gradient_checkpointing=True requires a model implementing "
                "gradient_checkpointing_enable()"
            )
        enabler()

    targets = resolve_target_modules(model, resolved)
    peft_type = resolved.peft_type or "lora"

    if peft_type == "prefix":
        lora_config = peft.PrefixTuningConfig(
            num_virtual_tokens=resolved.r,
            task_type=resolved.task_type,
        )
    elif peft_type == "adalora":
        lora_config = peft.AdaLoraConfig(
            r=resolved.r,
            lora_alpha=resolved.alpha,
            lora_dropout=resolved.dropout,
            target_modules=list(targets) if targets != "all-linear" else "all-linear",
            bias=resolved.bias,
            task_type=resolved.task_type,
            modules_to_save=list(resolved.modules_to_save) or None,
            init_lora_weights=resolved.init_lora_weights,
        )
    else:
        lora_config = peft.LoraConfig(
            r=resolved.r,
            lora_alpha=resolved.alpha,
            lora_dropout=resolved.dropout,
            target_modules=list(targets) if targets != "all-linear" else "all-linear",
            bias=resolved.bias,
            task_type=resolved.task_type,
            modules_to_save=list(resolved.modules_to_save) or None,
            init_lora_weights=resolved.init_lora_weights,
        )

    wrapped = peft.get_peft_model(model, lora_config)
    report = trainable_report(
        wrapped,
        adapter_names=tuple(str(name) for name in getattr(wrapped, "peft_config", {})),
        quantized=resolved.quantized,
    )
    report["peft_type"] = peft_type
    report["target_modules"] = (
        list(targets) if isinstance(targets, tuple) else targets
    )
    report["config"] = resolved.as_dict()
    return wrapped, report


def adapter_state(model: nn.Module) -> dict[str, Any]:
    """The adapter-only state dict, which is what a checkpoint stores for PEFT.

    A 4-bit base cannot be re-serialized, so the checkpoint keeps adapters and the
    base is reloaded from its own path on resume (document 09 section 4).
    """
    names = tuple(str(name) for name in getattr(model, "peft_config", {}))
    if not names:
        return {}
    state = getattr(model, "state_dict", None)
    if state is None:
        return {}
    return {name: value for name, value in state().items() if any(n in name for n in names)}


def merge_adapters(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base in place, when PEFT supports it."""
    merger = getattr(model, "merge_and_unload", None)
    if merger is None:
        raise ImproperlyConfigured(
            f"{type(model).__name__} has no merge_and_unload(); it is not a LoRA model"
        )
    return merger()
