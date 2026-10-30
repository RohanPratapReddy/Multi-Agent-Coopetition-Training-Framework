"""``doctor`` - report environment, runtime, and sandbox capability. Document 16, section 8.8.

It is what an operator runs when something is already wrong, so it works with settings
that do not load, a ``torch`` that cannot be imported, and no GPU: each probe becomes a
field in the report instead of a crash. ``torch`` and ``mactf.runtime`` are imported inside
the handler so ``help`` and completion never pay for them.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Any

from ..base import BaseCommand
from ..exitcodes import ExitCode


def _import_torch() -> Any:
    """``torch``, or ``None``. A broken CUDA install raises ``OSError``: a finding, not a crash."""
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001 - every failure to import is "torch is unusable"
        return None


def _project_info() -> dict[str, Any] | None:
    module = os.environ.get("MACTF_SETTINGS_MODULE")
    root = os.environ.get("MACTF_DISCOVERED_ROOT")
    if not root:
        return None
    return {"module": module, "root": root}


class Command(BaseCommand):
    help = "Diagnose the environment: Python, torch, GPU, and sandbox isolation."
    category = "Project"
    requires_settings = False
    output_formats = ("text", "json")
    examples = ("mactf doctor", "mactf doctor --format json", "mactf doctor --require cuda --require qemu")
    arguments = [
        (("--require",), {"action": "append", "choices": ["cuda", "qemu", "hardware-isolation"],
                          "default": None, "help": "Fail (exit 5) when this capability is missing."}),
    ]

    def handle(self, **options):
        import mactf

        from ...sandbox.base import IsolationLevel
        from ...sandbox.detect import probe_all

        payload: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()}",
            "executable": sys.executable,
        }
        settings = self.settings if self.settings_error is None else None
        torch_module = _import_torch()
        gpus: list[dict[str, Any]] = []
        if torch_module is not None:
            from ...runtime import resolve_device, resolve_dtype

            device = resolve_device(settings["DEVICE"] if settings else "auto")
            dtype = resolve_dtype(settings["DTYPE"] if settings else "auto", device)
            cuda = bool(torch_module.cuda.is_available())
            payload.update({
                "torch": torch_module.__version__, "cuda_available": cuda,
                "cuda_version": torch_module.version.cuda, "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "amp": bool(settings and settings["AMP"]) and device.type == "cuda",
            })
            for index in range(torch_module.cuda.device_count() if cuda else 0):
                props = torch_module.cuda.get_device_properties(index)
                gpus.append({"index": index, "name": props.name,
                             "total_memory_bytes": int(props.total_memory)})
        else:
            payload.update({"torch": None, "cuda_available": False, "cuda_version": None,
                            "device": None, "dtype": None, "amp": False})

        reports = probe_all()
        best = max((report.isolation for report in reports if report.usable),
                   default=IsolationLevel.NONE)
        payload.update({
            "qemu": shutil.which("qemu-system-x86_64") or shutil.which("qemu-system-x86_64.exe"),
            "sandbox": {report.provider: report.as_dict() for report in reports},
            "isolation_available": best.label,
            "settings_module": settings.module if settings else None,
            "mactf": mactf.__version__,
            "gpus": gpus,
            "settings_error": str(self.settings_error) if self.settings_error else None,
            "project": _project_info(),
        })
        met = {"cuda": payload["cuda_available"], "qemu": bool(payload["qemu"]),
               "hardware-isolation": best >= IsolationLevel.HARDWARE}
        wanted = options["require"] or []
        payload["requirements"] = [{"name": name, "met": met[name]} for name in wanted]
        unmet = [name for name in wanted if not met[name]]
        payload["ok"] = not unmet and self.settings_error is None
        self.out.emit(payload, text=lambda: self._text(payload, reports))
        if unmet:
            self.out.error("missing: " + ", ".join(unmet))
            return int(ExitCode.UNAVAILABLE)
        return int(ExitCode.FAILURE) if self.settings_error is not None else int(ExitCode.OK)

    def _text(self, payload: dict[str, Any], reports: Any) -> None:
        out = self.out
        out.result("MACTF doctor")
        out.result(f"  python      : {payload['python']} ({payload['platform']})")
        if payload["torch"] is None:
            out.result("  torch       : not importable (cuda=False)")
        else:
            out.result(f"  torch       : {payload['torch']} (cuda={payload['cuda_available']})")
            out.result(f"  device      : {payload['device']} dtype={payload['dtype']} amp={payload['amp']}")
        for gpu in payload["gpus"]:
            out.result(f"  gpu[{gpu['index']}]     : {gpu['name']} {gpu['total_memory_bytes'] / 1e9:.1f} GB")
        if payload["settings_error"]:
            out.result(f"  settings    : ERROR {payload['settings_error']}")
        project = payload["project"]
        if project:
            out.result(f"  project     : {project['module'] or '(framework defaults)'} at {project['root']}")
        out.result(f"  qemu        : {payload['qemu'] or 'not installed'}")
        for report in reports:
            out.result(f"  {report.describe()}")
        out.result(f"  isolation   : best available = {payload['isolation_available']}")
        if not payload["qemu"]:
            out.result(
                f"  hint        : install QEMU with `{self.prog} sandbox install` for hardware "
                "isolation, or keep using the local provider for development"
            )
