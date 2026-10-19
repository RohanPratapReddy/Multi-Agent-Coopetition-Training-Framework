"""``sandbox`` - probe, provision, install, and launch isolation backends."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..base import BaseCommand
from ...conf import get_settings
from ...exceptions import CommandError
from ...sandbox.detect import build_spec, choose_provider, probe_all


class Command(BaseCommand):
    help = "Manage the QEMU hardware sandbox and inspect isolation capability."
    arguments = [
        (("action",), {"nargs": "?", "default": "status",
                       "choices": ["status", "install", "provision", "launch", "clean"]}),
        (("--peer",), {"default": "swarm"}),
        (("--isolation",), {"default": None, "choices": ["none", "process", "container", "hardware"]}),
        (("--no-fallback",), {"action": "store_true", "default": False}),
        (("--json",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        action = options["action"]
        if action == "status":
            return self._status(options)
        if action == "install":
            return self._install()
        if action == "provision":
            return self._provision(options)
        if action == "launch":
            return self._launch(options)
        return self._clean(options)

    # -- actions ---------------------------------------------------------
    def _status(self, options) -> int:
        reports = probe_all()
        if options.get("json"):
            import json

            self.stdout.write(json.dumps([report.as_dict() for report in reports], indent=2) + "\n")
            return 0
        for report in reports:
            self.success(report.describe())
        usable = [report for report in reports if report.usable]
        best = max(usable, key=lambda report: int(report.isolation)) if usable else None
        self.success(f"best usable isolation: {best.isolation.label if best else 'none'}")
        return 0

    def _install(self) -> int:
        if sys.platform.startswith("win"):
            winget = shutil.which("winget")
            if not winget:
                raise CommandError(
                    "winget is not available. Install QEMU from https://qemu.weilnetz.de/ "
                    "and add it to PATH."
                )
            self.success("Installing QEMU via winget (this may require elevation)...")
            result = subprocess.run(  # noqa: S603 - fixed source, explicit args
                [
                    winget,
                    "install",
                    "-e",
                    "--id",
                    "SoftwareFreedomConservancy.QEMU",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                check=False,
            )
            if result.returncode != 0:
                raise CommandError(f"winget exited with {result.returncode}")
            self.success("QEMU installed. WHPX acceleration needs the Windows Hypervisor Platform:")
            self.success("  (admin) dism /online /enable-feature /featurename:HypervisorPlatform /all")
            self.success("  then reboot and re-run `manage.py sandbox status`")
            return 0

        if sys.platform.startswith("linux"):
            managers = [
                ("apt-get", ["apt-get", "install", "-y", "qemu-system-x86", "qemu-utils"]),
                ("dnf", ["dnf", "install", "-y", "qemu-kvm", "qemu-img"]),
                ("pacman", ["pacman", "-S", "--noconfirm", "qemu-full"]),
            ]
            for binary, args in managers:
                if shutil.which(binary):
                    self.success(f"Installing QEMU via {binary}: {' '.join(args)}")
                    subprocess.run(args, check=False)  # noqa: S603
                    self.success("Verify KVM access with: ls -l /dev/kvm")
                    return 0
            raise CommandError("no supported package manager found (apt/dnf/pacman)")

        raise CommandError(
            "automatic install is supported on Windows (winget) and Linux only; "
            "install QEMU manually on this platform"
        )

    def _provision(self, options) -> int:
        settings = get_settings()
        choice = choose_provider(
            requested=str(settings["SANDBOX"].get("PROVIDER", "auto")),
            isolation=options.get("isolation") or settings["SANDBOX"].get("ISOLATION"),
            settings=settings["SANDBOX"],
            allow_fallback=not options.get("no_fallback"),
        )
        self.success(choice.describe())
        if not choice.report.usable:
            raise CommandError("selected provider is not usable")
        spec = build_spec(
            options["peer"], settings=settings["SANDBOX"], root=settings.path("ROOT")
        )
        self.success(f"spec: {spec.describe()}")
        if spec.base_image is None or not Path(spec.base_image).exists():
            self.success(
                f"base image missing: {spec.base_image}\n"
                "Download a cloud image, for example:\n"
                "  https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img"
            )
            return 1
        handle = choice.provider.provision(spec)
        self.success(f"provisioned: {handle.describe()}")
        return 0

    def _launch(self, options) -> int:
        settings = get_settings()
        choice = choose_provider(
            requested=str(settings["SANDBOX"].get("PROVIDER", "auto")),
            isolation=options.get("isolation") or settings["SANDBOX"].get("ISOLATION"),
            settings=settings["SANDBOX"],
            allow_fallback=not options.get("no_fallback"),
        )
        spec = build_spec(
            options["peer"], settings=settings["SANDBOX"], root=settings.path("ROOT")
        )
        handle = choice.provider.provision(spec)
        self.success(f"launching: {handle.describe()}")
        handle = choice.provider.start(handle)
        self.success(f"ready: pid={handle.pid} endpoints={handle.endpoints}")
        self.success("use `manage.py sandbox clean` to remove overlay disks")
        return 0

    def _clean(self, options) -> int:
        settings = get_settings()
        root = settings.path("ROOT")
        removed = 0
        for pattern in ("peer-*.qcow2", "*.sock"):
            for path in root.glob(f"var/**/{pattern}"):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        self.success(f"removed {removed} sandbox artifact(s)")
        return 0
