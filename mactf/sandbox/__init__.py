"""Sandbox package: hardware-isolated execution with capability reporting."""

from __future__ import annotations

from .base import (
    Capability,
    CapabilityReport,
    ExecResult,
    IsolationLevel,
    MissingCapability,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    ensure_hardware,
)
from .detect import ProviderChoice, build_spec, candidate_providers, choose_provider, probe_all
from .local import LocalSandbox
from .provision import CloudInitSeed, OverlayResult, create_overlay, write_cloud_init
from .qemu import QemuSandbox

__all__ = [
    "Capability",
    "CapabilityReport",
    "CloudInitSeed",
    "ExecResult",
    "IsolationLevel",
    "LocalSandbox",
    "MissingCapability",
    "OverlayResult",
    "ProviderChoice",
    "QemuSandbox",
    "SandboxHandle",
    "SandboxProvider",
    "SandboxSpec",
    "build_spec",
    "candidate_providers",
    "choose_provider",
    "create_overlay",
    "ensure_hardware",
    "probe_all",
    "write_cloud_init",
]
