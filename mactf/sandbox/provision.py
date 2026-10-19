"""Provisioning artifacts for QEMU sandboxes.

Cloud-init delivery and overlay disk management, kept separate from the provider
so the same logic is testable without launching a VM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import SandboxLaunchError

WINDOWS = sys.platform.startswith("win")


def image_binary() -> str:
    return "qemu-img.exe" if WINDOWS else "qemu-img"


def find_image_tool() -> str | None:
    return shutil.which(image_binary())


@dataclass
class OverlayResult:
    path: Path
    base: Path
    created: bool
    command: list[str]

    def describe(self) -> str:
        action = "created" if self.created else "reused"
        return f"{action} overlay {self.path.name} from {self.base.name}"


def create_overlay(
    base: Path | str,
    work: Path | str,
    *,
    resize_gb: int = 0,
    force: bool = False,
    timeout: float = 120.0,
) -> OverlayResult:
    """Create (or reuse) a qcow2 copy-on-write overlay backed by ``base``."""
    base_path = Path(base)
    work_path = Path(work)
    if not base_path.exists():
        raise SandboxLaunchError(f"base image not found: {base_path}")
    tool = find_image_tool()
    if tool is None:
        raise SandboxLaunchError("qemu-img is required to build a copy-on-write disk")

    stale = force or not work_path.exists()
    if not stale:
        try:
            stale = work_path.stat().st_mtime < base_path.stat().st_mtime
        except OSError:
            stale = True

    if not stale:
        return OverlayResult(path=work_path, base=base_path, created=False, command=[])

    work_path.parent.mkdir(parents=True, exist_ok=True)
    work_path.unlink(missing_ok=True)
    command = [
        tool,
        "create",
        "-f",
        "qcow2",
        "-F",
        "qcow2",
        "-b",
        str(base_path.resolve()),
        str(work_path),
    ]
    completed = subprocess.run(  # noqa: S603 - fixed binary, explicit args
        command, capture_output=True, text=True, timeout=timeout, shell=False
    )
    if completed.returncode != 0:
        raise SandboxLaunchError(
            f"qemu-img create failed: {(completed.stderr or completed.stdout).strip()}"
        )

    if resize_gb > 0:
        resize = [tool, "resize", str(work_path), f"{int(resize_gb)}G"]
        result = subprocess.run(  # noqa: S603
            resize, capture_output=True, text=True, timeout=timeout, shell=False
        )
        if result.returncode != 0:
            raise SandboxLaunchError(
                f"qemu-img resize failed: {(result.stderr or result.stdout).strip()}"
            )

    return OverlayResult(path=work_path, base=base_path, created=True, command=command)


@dataclass
class CloudInitSeed:
    directory: Path
    user_data: Path
    meta_data: Path
    vendor_data: Path
    fw_cfg_payload: Path

    def describe(self) -> str:
        return f"cloud-init seed at {self.directory}"


def write_cloud_init(
    directory: Path | str,
    *,
    peer: str,
    user: str = "mactf",
    packages: list[str] | None = None,
    ssh_authorized_keys: list[str] | None = None,
) -> CloudInitSeed:
    """Write a NoCloud seed directory usable through QEMU's ``fw_cfg``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    packages = packages or ["python3", "python3-pip"]
    lines = [
        "#cloud-config",
        "package_update: true",
        "packages:",
        *[f"  - {package}" for package in packages],
        "users:",
        f"  - name: {user}",
        "    sudo: ALL=(ALL) NOPASSWD:ALL",
        "    shell: /bin/bash",
    ]
    if ssh_authorized_keys:
        lines.append("    ssh_authorized_keys:")
        lines.extend(f"      - {key}" for key in ssh_authorized_keys)
    lines.extend(
        [
            "runcmd:",
            "  - [ sh, -c, 'echo mactf-ready > /run/mactf.ready' ]",
            "",
        ]
    )
    user_data = "\n".join(lines)
    meta_data = f"instance-id: mactf-{peer}\nlocal-hostname: mactf-{peer}\n"

    (target / "user-data").write_text(user_data, encoding="utf-8")
    (target / "meta-data").write_text(meta_data, encoding="utf-8")
    (target / "vendor-data").write_text("#cloud-config\n", encoding="utf-8")

    payload = {
        "instance-id": f"mactf-{peer}",
        "local-hostname": f"mactf-{peer}",
        "user-data": user_data,
        "meta-data": meta_data,
    }
    fw_cfg = target / "cloudinit.json"
    fw_cfg.write_text(json.dumps(payload), encoding="utf-8")

    return CloudInitSeed(
        directory=target,
        user_data=target / "user-data",
        meta_data=target / "meta-data",
        vendor_data=target / "vendor-data",
        fw_cfg_payload=fw_cfg,
    )


def seed_payload(directory: Path | str) -> dict[str, Any]:
    """Read the fw_cfg payload back, raising a clear error when incomplete."""
    path = Path(directory) / "cloudinit.json"
    if not path.exists():
        raise SandboxLaunchError(f"cloud-init payload missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def temporary_seed(**kwargs: Any) -> CloudInitSeed:
    directory = Path(tempfile.mkdtemp(prefix="mactf-seed-"))
    return write_cloud_init(directory, **kwargs)
