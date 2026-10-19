"""A subprocess sandbox for hosts without hardware virtualization.

This provider exists so the framework is usable and testable everywhere. It is
explicitly *not* hardware isolation: it reports ``PROCESS`` level and records
why. Any run that requires hardware isolation must refuse it, which is the
framework.md P20 discipline applied to the tooling itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .base import (
    Capability,
    CapabilityReport,
    ExecResult,
    IsolationLevel,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
)

_SHELL_BINARIES = ("bash", "sh")


class LocalSandbox(SandboxProvider):
    """Process-level sandbox using a temp workspace and a restricted environment."""

    name = "local"
    isolation = IsolationLevel.PROCESS

    def __init__(self, *, workdir: Path | None = None) -> None:
        self._root = Path(workdir or tempfile.mkdtemp(prefix="mactf-local-"))
        self._handles: dict[str, SandboxHandle] = {}

    # -- probing ---------------------------------------------------------
    def probe(self) -> CapabilityReport:
        shell = next((path for path in (shutil.which(name) for name in _SHELL_BINARIES) if path), None)
        capabilities = [
            Capability("python", True, sys.executable),
            Capability("shell", bool(shell), shell or "none"),
            Capability("hardware-isolation", False, "process-level only"),
        ]
        reasons = [
            "no hardware virtualization: framework.md P20 containment cannot be "
            "guaranteed, so treat escapes as unverifiable and log this provider"
        ]
        try:
            writable = self._root.exists() or self._root.mkdir(parents=True, exist_ok=True) is None
        except OSError:
            writable = False
        capabilities.append(Capability("writable-workdir", writable, str(self._root)))
        if not writable:
            reasons.append(f"workspace {self._root} is not writable")
        return CapabilityReport(
            provider=self.name,
            isolation=self.isolation,
            usable=bool(writable),
            capabilities=capabilities,
            reasons=reasons,
            metadata={"root": str(self._root)},
        )

    # -- lifecycle -------------------------------------------------------
    def provision(self, spec: SandboxSpec) -> SandboxHandle:
        workspace = self._root / spec.peer
        workspace.mkdir(parents=True, exist_ok=True)
        handle = SandboxHandle(
            provider=self.name,
            peer=spec.peer,
            isolation=self.isolation,
            spec=spec,
            metadata={"workspace": str(workspace)},
        )
        return handle

    def start(self, handle: SandboxHandle) -> SandboxHandle:
        handle.metadata.setdefault("started_at", time.time())
        handle.pid = os.getpid()
        handle.endpoints["workspace"] = handle.metadata["workspace"]
        self._handles[handle.peer] = handle
        return handle

    def exec(
        self, handle: SandboxHandle, command: list[str] | str, *, timeout: float | None = None
    ) -> ExecResult:
        workspace = Path(handle.metadata["workspace"])
        if isinstance(command, str):
            shell = next((shutil.which(name) for name in _SHELL_BINARIES if shutil.which(name)), None)
            if shell is None:
                return ExecResult(
                    command=command,
                    stderr="no shell available to run a string command",
                    exit_code=127,
                )
            args = [shell, "-c", command]
            use_shell = False
        else:
            args = list(command)
            use_shell = False

        # A deliberately small environment: no credentials, no host paths.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(workspace),
            "TMPDIR": str(workspace),
            "MACTF_SANDBOX": self.name,
            "MACTF_ISOLATION": self.isolation.label,
        }
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - explicit args, no shell injection
                args,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or handle.spec.command_timeout,
                shell=use_shell,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                command=command,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "timed out",
                exit_code=124,
                duration=time.monotonic() - started,
                timed_out=True,
            )
        except OSError as exc:
            return ExecResult(
                command=command,
                stderr=str(exc),
                exit_code=127,
                duration=time.monotonic() - started,
            )
        return ExecResult(
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            duration=time.monotonic() - started,
        )

    def stop(self, handle: SandboxHandle, *, force: bool = False) -> None:
        self._handles.pop(handle.peer, None)
        handle.pid = None

    def cleanup(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
