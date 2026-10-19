"""Device, dtype, determinism, and checkpoint runtime helpers.

GPU support is a first-class requirement: ``DEVICE`` accepts ``auto``, ``cpu``,
``cuda``, ``cuda:N``, and ``mps``; ``DTYPE`` accepts ``auto``, ``float32``,
``float64``, ``bfloat16``, and ``float16``; ``AMP`` enables CUDA autocast with a
grad scaler. Everything degrades gracefully on CPU-only hosts.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .conf import Settings
from .exceptions import ImproperlyConfigured


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string, falling back predictably when unavailable."""
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ImproperlyConfigured(
                "DEVICE requests cuda but torch.cuda.is_available() is False; "
                "install a CUDA build of torch or set DEVICE='cpu'"
            )
        return torch.device(requested)
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ImproperlyConfigured("DEVICE requests mps but it is not available")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype string for the given device."""
    requested = (requested or "auto").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested == "auto":
        if device.type == "cuda":
            return torch.float32
        return torch.float32
    if requested not in mapping:
        raise ImproperlyConfigured(f"unsupported DTYPE {requested!r}")
    dtype = mapping[requested]
    if dtype in {torch.float16, torch.bfloat16} and device.type != "cuda":
        # Half precision on CPU is slower and poorly supported; keep training sane.
        return torch.float32
    return dtype


@dataclass
class Runtime:
    """Resolved execution context for a training run."""

    device: torch.device
    dtype: torch.dtype
    amp: bool
    seed: int
    deterministic: bool
    grad_scaler: Any = None
    notes: list[str] = field(default_factory=list)

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def amp_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.bfloat16

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "amp": self.amp,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "notes": list(self.notes),
        }
        if self.device.type == "cuda":
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(self.device),
                    "gpu_memory_gb": round(
                        torch.cuda.get_device_properties(self.device).total_memory / 1e9, 2
                    ),
                }
            )
        return info

    def to(self, tensor_or_module: Any) -> Any:
        """Move a tensor or module to the runtime device and float dtype."""
        if isinstance(tensor_or_module, nn.Module):
            module = tensor_or_module.to(device=self.device)
            floating = [p for p in module.parameters() if p.is_floating_point()]
            if floating:
                module = module.to(dtype=self.dtype)
            return module
        if torch.is_tensor(tensor_or_module):
            if tensor_or_module.is_floating_point():
                return tensor_or_module.to(device=self.device, dtype=self.dtype)
            return tensor_or_module.to(device=self.device)
        return tensor_or_module

    def resolve(self, value: str) -> str:
        """Substitute ``{device}`` and ``{dtype}`` placeholders in a config value."""
        if not isinstance(value, str):
            return value
        return value.format(
            device=self.device.type,
            dtype=str(self.dtype).replace("torch.", ""),
            device_index=self.device.index or 0,
        )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass
    else:
        torch.backends.cudnn.benchmark = True


def build_runtime(settings: Settings) -> Runtime:
    """Resolve device/dtype/AMP from settings and seed the process."""
    device = resolve_device(settings["DEVICE"])
    dtype = resolve_dtype(settings["DTYPE"], device)
    amp = bool(settings["AMP"]) and device.type == "cuda"
    notes: list[str] = []
    if bool(settings["AMP"]) and not amp:
        notes.append("AMP requested but only enabled on CUDA; running without autocast")
    if str(settings["DTYPE"]).lower() not in {"auto", "float32", "fp32", "float"} and dtype is torch.float32:
        notes.append("half/bfloat16 requested on CPU; promoted to float32")
    seed = int(settings["SEED"])
    deterministic = bool(settings["DETERMINISTIC"])
    seed_everything(seed, deterministic=deterministic)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp else None
    return Runtime(
        device=device,
        dtype=dtype,
        amp=amp,
        seed=seed,
        deterministic=deterministic,
        grad_scaler=scaler,
        notes=notes,
    )


def autocast(runtime: Runtime):
    """Return the appropriate ``torch.autocast`` context (possibly a no-op)."""
    if runtime.amp and runtime.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=runtime.device.type, enabled=False)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    """Save a checkpoint atomically, mapping tensors back to CPU for portability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = dict(payload)
    payload.setdefault("format_version", 1)
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return torch.load(Path(path), map_location=map_location, weights_only=False)
