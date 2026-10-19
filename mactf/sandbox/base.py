"""Sandbox abstraction: capability reporting, specs, and the provider contract.

Isolation is a scored property, not a deployment detail: framework.md P20 and
P21 depend on the boundary being real. Every provider therefore reports exactly
what it can and cannot contain, and the orchestrator records that report.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any


class IsolationLevel(IntEnum):
    """Ordered isolation strength."""

    NONE = 0
    PROCESS = 1
    CONTAINER = 2
    HARDWARE = 3

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: Any) -> "IsolationLevel":
        if isinstance(value, IsolationLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip().upper()
        if text.isdigit():
            return cls(int(text))
        try:
            return cls[text]
        except KeyError as exc:
            raise ValueError(
                f"unknown isolation level {value!r}; expected one of "
                f"{[level.name for level in cls]}"
            ) from exc


@dataclass(frozen=True)
class Capability:
    """One probe result."""

    name: str
    available: bool
    detail: str = ""

    def describe(self) -> str:
        state = "ok" if self.available else "missing"
        return f"{self.name}={state}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class CapabilityReport:
    """What a provider found on this host."""

    provider: str
    isolation: IsolationLevel
    usable: bool
    capabilities: list[Capability] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Capability | None:
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None

    def require(self, *names: str) -> None:
        missing = [
            name
            for name in names
            if (self.get(name) is None or not self.get(name).available)  # type: ignore[union-attr]
        ]
        if missing:
            raise MissingCapability(
                f"{self.provider} is missing required capability: {', '.join(missing)}"
            )

    def describe(self) -> str:
        caps = ", ".join(capability.describe() for capability in self.capabilities)
        state = "usable" if self.usable else "unusable"
        reasons = f" reasons=[{'; '.join(self.reasons)}]" if self.reasons else ""
        return f"{self.provider} ({self.isolation.label}) {state} [{caps}]{reasons}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "isolation": self.isolation.label,
            "usable": self.usable,
            "capabilities": {
                capability.name: capability.available for capability in self.capabilities
            },
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


class MissingCapability(Exception):
    """Raised when a provider cannot meet a requested isolation guarantee."""


@dataclass
class SandboxSpec:
    """How a peer's sandbox should be provisioned and launched."""

    peer: str
    isolation: IsolationLevel = IsolationLevel.HARDWARE
    memory_mb: int = 4096
    vcpus: int = 2
    network: str = "none"
    snapshot: bool = True
    base_image: Path | None = None
    work_image: Path | None = None
    accel: str = "auto"
    boot_timeout: float = 180.0
    command_timeout: float = 300.0
    labels: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        accel = self.accel
        return (
            f"spec[{self.peer}] isolation={self.isolation.label} "
            f"mem={self.memory_mb}MB vcpus={self.vcpus} net={self.network} "
            f"snapshot={self.snapshot} accel={accel}"
        )


@dataclass
class ExecResult:
    """The outcome of one command executed inside a sandbox."""

    command: list[str] | str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def describe(self) -> str:
        status = "ok" if self.ok else f"failed({self.exit_code})"
        return f"exec {status} in {self.duration:.2f}s"

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration": self.duration,
            "timed_out": self.timed_out,
        }


@dataclass
class SandboxHandle:
    """A live or ready sandbox instance."""

    provider: str
    peer: str
    isolation: IsolationLevel
    spec: SandboxSpec
    pid: int | None = None
    endpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.provider}:{self.peer} ({self.isolation.label}) "
            f"pid={self.pid} endpoints={list(self.endpoints)}"
        )


class SandboxProvider(abc.ABC):
    """Base class for isolation backends.

    ``probe`` must never raise for a missing dependency; it reports instead, so a
    training run degrades predictably and can record the reason.
    """

    name: str = "base"
    isolation: IsolationLevel = IsolationLevel.NONE

    @abc.abstractmethod
    def probe(self) -> CapabilityReport:
        """Inspect the host and report what this provider can do."""

    @abc.abstractmethod
    def provision(self, spec: SandboxSpec) -> SandboxHandle:
        """Prepare the sandbox image or workspace. Idempotent."""

    @abc.abstractmethod
    def start(self, handle: SandboxHandle) -> SandboxHandle:
        """Boot the sandbox and wait for readiness."""

    @abc.abstractmethod
    def exec(
        self, handle: SandboxHandle, command: list[str] | str, *, timeout: float | None = None
    ) -> ExecResult:
        """Run a command inside the sandbox."""

    @abc.abstractmethod
    def stop(self, handle: SandboxHandle, *, force: bool = False) -> None:
        """Shut the sandbox down and release resources."""

    def __enter__(self) -> "SandboxProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def describe(self) -> str:
        return f"<{type(self).__name__} name={self.name} isolation={self.isolation.label}>"


def ensure_hardware(report: CapabilityReport, *, allow_fallback: bool) -> None:
    """Enforce a hardware-isolation requirement when the caller demands it."""
    if report.isolation >= IsolationLevel.HARDWARE:
        return
    if not allow_fallback:
        raise MissingCapability(
            f"hardware isolation required but provider {report.provider!r} offers "
            f"{report.isolation.label}: {'; '.join(report.reasons) or 'no reason given'}"
        )
