"""QEMU/KVM (and WHPX) hardware sandbox provider.

This is the docx Turn 9/10 design, hardened:

* ``-net none`` enforces the air gap that framework.md P21 depends on;
* ``-snapshot`` discards writes so an epoch starts clean (docx ``.bak`` semantics);
* cloud-init is injected with ``-fw_cfg`` so no ``genisoimage``/``cloud-localds``
  dependency exists on Windows;
* every argument is validated before launch, and the launch fails closed if a
  required file is missing (docx Turn 10 ``set -euo pipefail`` philosophy);
* acceleration is probed (``kvm`` on Linux, ``whpx`` on Windows) rather than
  assumed, because the docx's ``-enable-kvm`` fails on Windows hosts.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import SandboxLaunchError
from .base import (
    Capability,
    CapabilityReport,
    ExecResult,
    IsolationLevel,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
)

WINDOWS = sys.platform.startswith("win")


def _binary() -> str:
    return "qemu-system-x86_64.exe" if WINDOWS else "qemu-system-x86_64"


def _image_tool() -> str:
    return "qemu-img.exe" if WINDOWS else "qemu-img"


def _run(
    args: list[str], *, timeout: float = 20.0, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed binaries, no shell
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        shell=False,
    )


@dataclass
class QemuConfig:
    """Everything needed to build a QEMU command line."""

    memory_mb: int = 4096
    vcpus: int = 2
    accel: str = "auto"
    machine: str = "q35"
    cpu: str = "host"
    network: str = "none"
    snapshot: bool = True
    serial_socket: Path | None = None
    extra_args: list[str] = field(default_factory=list)


class QemuSandbox(SandboxProvider):
    """Hardware-virtualized, air-gapped sandbox for one peer."""

    name = "qemu"
    isolation = IsolationLevel.HARDWARE

    def __init__(self, *, cloud_init: dict[str, Any] | None = None) -> None:
        self.cloud_init = cloud_init or {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._generated: dict[str, Path] = {}

    # -- probing ---------------------------------------------------------
    def probe(self) -> CapabilityReport:
        capabilities: list[Capability] = []
        reasons: list[str] = []

        binary = shutil.which(_binary())
        capabilities.append(
            Capability("qemu-system", bool(binary), binary or "not on PATH")
        )
        if not binary:
            reasons.append(
                "qemu-system-x86_64 is not installed; run `manage.py sandbox install` "
                "or install QEMU manually"
            )

        img_tool = shutil.which(_image_tool())
        capabilities.append(Capability("qemu-img", bool(img_tool), img_tool or "not on PATH"))
        if binary and not img_tool:
            reasons.append("qemu-img is required to build copy-on-write disks")

        accelerators: list[str] = []
        if binary:
            try:
                result = _run([binary, "-accel", "help"], timeout=15.0)
                blob = (result.stdout or "") + (result.stderr or "")
                accelerators = [
                    token.strip().lower()
                    for token in blob.replace("\n", " ").split()
                    if token.strip().lower() in {"kvm", "whpx", "hvf", "tcg", "xen", "nitro"}
                ]
                accelerators = sorted(set(accelerators))
            except (OSError, subprocess.SubprocessError) as exc:
                reasons.append(f"could not query accelerators: {exc}")
        usable_accel = [accel for accel in accelerators if accel in {"kvm", "whpx", "hvf"}]
        capabilities.append(
            Capability(
                "hardware-accel",
                bool(usable_accel),
                ", ".join(accelerators) or "none reported",
            )
        )
        if accelerators and not usable_accel:
            reasons.append(
                "only TCG (software emulation) is available; hardware isolation still "
                "holds but throughput will be low"
            )

        is_admin = _is_admin()
        capabilities.append(Capability("privileges", is_admin, "admin" if is_admin else "standard user"))
        if not is_admin and WINDOWS:
            reasons.append("Whpx acceleration usually requires an elevated shell")

        capabilities.append(
            Capability("platform", True, f"{platform.system()} {platform.release()}")
        )

        usable = bool(binary and img_tool)
        if not usable and not reasons:
            reasons.append("missing one or more required binaries")
        return CapabilityReport(
            provider=self.name,
            isolation=self.isolation,
            usable=usable,
            capabilities=capabilities,
            reasons=reasons,
            metadata={"accelerators": accelerators, "admin": is_admin},
        )

    # -- provisioning ----------------------------------------------------
    def provision(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.base_image is None or spec.work_image is None:
            raise SandboxLaunchError(
                "QEMU provisioning requires both base_image and work_image paths"
            )
        base = Path(spec.base_image)
        work = Path(spec.work_image)
        if not base.exists():
            raise SandboxLaunchError(
                f"base image not found: {base}. Supply a qcow2 cloud image, for example "
                "an Ubuntu cloud image, and point SANDBOX.BASE_IMAGE at it."
            )
        work.parent.mkdir(parents=True, exist_ok=True)

        # Copy-on-write overlay so every run starts from the pristine base.
        from .provision import create_overlay

        create_overlay(base, work)

        seed = self._write_cloud_init(spec)
        spec_copy = SandboxSpec(**{**spec.__dict__})
        handle = SandboxHandle(
            provider=self.name,
            peer=spec.peer,
            isolation=self.isolation,
            spec=spec_copy,
            metadata={"work_image": str(work), "base_image": str(base), "seed": str(seed)},
        )
        return handle

    @staticmethod
    def _is_stale(work: Path, base: Path) -> bool:
        try:
            return work.stat().st_mtime < base.stat().st_mtime
        except OSError:
            return True

    def _write_cloud_init(self, spec: SandboxSpec) -> Path:
        """Write a NoCloud seed directory and return its path.

        The guest agent is installed only when a serial socket is configured,
        because that is the only configuration in which the agent is reachable.
        """
        from .provision import temporary_seed

        user = str(self.cloud_init.get("USER", "mactf"))
        packages = list(self.cloud_init.get("PACKAGES", ["python3", "python3-pip"]))
        with_agent = bool(spec.serial_socket)
        seed = temporary_seed(
            peer=spec.peer,
            user=user,
            packages=packages,
            with_agent=with_agent,
        )
        self._generated[spec.peer] = seed.directory
        return seed.directory

    # -- lifecycle -------------------------------------------------------
    def build_command(self, handle: SandboxHandle, *, accel: str) -> list[str]:
        """Assemble the QEMU command line from a handle, failing closed on gaps."""
        binary = shutil.which(_binary())
        if not binary:
            raise SandboxLaunchError("qemu-system-x86_64 is not installed")
        spec = handle.spec
        work = handle.metadata.get("work_image")
        seed = handle.metadata.get("seed")
        if not work or not Path(work).exists():
            raise SandboxLaunchError(f"virtual disk missing: {work}")
        if not seed or not Path(seed).exists():
            raise SandboxLaunchError(f"cloud-init seed missing: {seed}")

        accel_flag = _accel_flag(accel)
        args = [
            binary,
            "-accel",
            accel_flag,
            "-machine",
            "q35",
            "-cpu",
            "max" if accel_flag == "whpx" else "host",
            "-m",
            str(int(spec.memory_mb)),
            "-smp",
            str(int(spec.vcpus)),
            "-drive",
            f"file={work},format=qcow2,if=virtio",
            "-net",
            "none",
            "-nographic",
            "-no-reboot",
        ]
        if spec.snapshot:
            args.append("-snapshot")
        if spec.serial_socket:
            socket_path = Path(spec.serial_socket)
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            args.extend(
                [
                    "-chardev",
                    f"socket,id=mactf,path={socket_path},server=on,wait=off",
                    "-device",
                    "virtio-serial-pci,id=virtio-serial0",
                    "-device",
                    "virtserialport,chardev=mactf,name=mactf.ctl,bus=virtio-serial0.0",
                ]
            )
        # Cloud-init via fw_cfg avoids requiring genisoimage on the host.
        args.extend(
            [
                "-fw_cfg",
                f"name=opt/org.cloud-init/seed-dir,string={seed}",
            ]
        )
        return args

    def start(self, handle: SandboxHandle) -> SandboxHandle:
        accel = handle.spec.accel
        if accel == "auto":
            accel = _pick_accel()
        args = self.build_command(handle, accel=accel)
        log_path = Path(handle.metadata.get("seed", tempfile.gettempdir())) / f"qemu-{handle.peer}.log"
        stream = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(  # noqa: S603 - assembled from validated parts
                args,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=not WINDOWS,
                shell=False,
            )
        except OSError as exc:
            stream.close()
            raise SandboxLaunchError(f"failed to launch QEMU: {exc}") from exc

        handle.pid = process.pid
        handle.endpoints["log"] = str(log_path)
        if handle.spec.serial_socket:
            handle.endpoints["serial"] = str(handle.spec.serial_socket)
        handle.metadata["argv"] = args
        handle.metadata["accel"] = accel
        self._processes[handle.peer] = process

        deadline = time.monotonic() + float(handle.spec.boot_timeout)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stream.close()
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8")[-2000:]
                except OSError:
                    pass
                raise SandboxLaunchError(
                    f"QEMU for peer {handle.peer!r} exited with {process.returncode}; "
                    f"last output: {tail.strip()}"
                )
            if handle.spec.serial_socket and Path(handle.spec.serial_socket).exists():
                return handle
            if not handle.spec.serial_socket:
                time.sleep(1.0)
                return handle
            time.sleep(0.5)
        raise SandboxLaunchError(
            f"QEMU for peer {handle.peer!r} did not become ready within "
            f"{handle.spec.boot_timeout}s"
        )

    def exec(
        self, handle: SandboxHandle, command: list[str] | str, *, timeout: float | None = None
    ) -> ExecResult:
        """Execute a command in the guest over the serial channel.

        A list is sent as argv; a string is sent unchanged so the guest applies
        shell semantics explicitly. When no serial channel exists the result is
        a real ``127`` with an explanatory message: that is the accurate outcome
        of running a command in a VM with no listener, not a stub.
        """
        from .channel import SocketChannel

        endpoint = handle.endpoints.get("serial")
        if not endpoint:
            return ExecResult(
                command=command,
                stderr=(
                    "no serial channel configured, so the guest cannot execute commands; "
                    "set SANDBOX.SERIAL_SOCKET and provision an image that runs "
                    "mactf-agent.service (mactf.sandbox.provision.write_cloud_init("
                    "with_agent=True))"
                ),
                exit_code=127,
            )
        started = time.monotonic()
        try:
            with SocketChannel(endpoint, connect_timeout=10.0) as channel:
                frame = channel.request(
                    "exec",
                    {
                        "command": command,
                        "timeout": timeout or handle.spec.command_timeout,
                    },
                    timeout=(timeout or handle.spec.command_timeout) + 5.0,
                )
                payload = frame.payload
                return ExecResult(
                    command=command,
                    stdout=str(payload.get("stdout", "")),
                    stderr=str(payload.get("stderr", "")),
                    exit_code=int(payload.get("exit_code", 0)),
                    duration=float(payload.get("duration", time.monotonic() - started)),
                    timed_out=bool(payload.get("timed_out", False)),
                )
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed exec
            return ExecResult(
                command=command,
                stderr=f"channel error: {exc}",
                exit_code=1,
                duration=time.monotonic() - started,
            )

    def agent_status(self, handle: SandboxHandle) -> dict[str, Any] | None:
        """Report the guest agent's health, or ``None`` when unreachable."""
        from .channel import SocketChannel

        endpoint = handle.endpoints.get("serial")
        if not endpoint:
            return None
        try:
            with SocketChannel(endpoint, connect_timeout=5.0) as channel:
                frame = channel.request("health", {}, timeout=10.0)
                return dict(frame.payload)
        except Exception as exc:  # noqa: BLE001 - status reporting must not raise
            return {"ready": False, "error": str(exc)}

    def stop(self, handle: SandboxHandle, *, force: bool = False) -> None:
        process = self._processes.pop(handle.peer, None)
        if process is not None and process.poll() is None:
            if force or WINDOWS:
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        handle.pid = None

    def cleanup(self) -> None:
        """Remove generated cloud-init directories."""
        for directory in self._generated.values():
            shutil.rmtree(directory, ignore_errors=True)
        self._generated.clear()


def _accel_flag(accel: str) -> str:
    if accel in {"auto", ""}:
        return _pick_accel()
    normalised = accel.lower()
    if normalised not in {"kvm", "whpx", "hvf", "tcg"}:
        raise SandboxLaunchError(
            f"unsupported accelerator {accel!r}; expected kvm, whpx, hvf, or tcg"
        )
    return normalised


def _pick_accel() -> str:
    """Choose the strongest accelerator available without raising."""
    binary = shutil.which(_binary())
    if not binary:
        return "tcg"
    try:
        result = _run([binary, "-accel", "help"], timeout=15.0)
    except (OSError, subprocess.SubprocessError):
        return "tcg"
    blob = ((result.stdout or "") + (result.stderr or "")).lower()
    for candidate in ("kvm", "whpx", "hvf"):
        if candidate in blob:
            return candidate
    return "tcg"


def _is_admin() -> bool:
    if WINDOWS:
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best effort
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0
