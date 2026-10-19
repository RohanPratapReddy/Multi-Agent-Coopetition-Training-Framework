"""``doctor`` - report environment, runtime, and sandbox capability."""

from __future__ import annotations

import platform
import shutil
import sys

from ..base import BaseCommand
from ...conf import get_settings
from ...runtime import resolve_device, resolve_dtype
from ...sandbox.detect import probe_all


class Command(BaseCommand):
    help = "Diagnose the environment: Python, torch, GPU, and sandbox isolation."
    arguments = [
        (("--json",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        import torch

        settings = get_settings()
        device = resolve_device(settings["DEVICE"])
        dtype = resolve_dtype(settings["DTYPE"], device)
        reports = probe_all()

        payload = {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()}",
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "amp": bool(settings["AMP"]) and device.type == "cuda",
            "qemu": shutil.which("qemu-system-x86_64") or shutil.which("qemu-system-x86_64.exe"),
            "sandbox": {report.provider: report.as_dict() for report in reports},
            "isolation_available": max(
                (report.isolation.label for report in reports if report.usable), default="none"
            ),
            "settings_module": settings.module,
        }

        if options.get("json"):
            import json

            self.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 0

        self.success("MACTF doctor")
        self.success(f"  python      : {payload['python']} ({payload['platform']})")
        self.success(f"  torch       : {payload['torch']} (cuda={payload['cuda_available']})")
        self.success(f"  device      : {payload['device']} dtype={payload['dtype']} amp={payload['amp']}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                self.success(
                    f"  gpu[{index}]     : {props.name} "
                    f"{props.total_memory / 1e9:.1f} GB"
                )
        self.success(f"  qemu        : {payload['qemu'] or 'not installed'}")
        for report in reports:
            self.success(f"  {report.describe()}")
        self.success(f"  isolation   : best available = {payload['isolation_available']}")
        if not payload["qemu"]:
            self.success(
                "  hint        : install QEMU with `manage.py sandbox install` for hardware "
                "isolation, or keep using the local provider for development"
            )
        return 0
